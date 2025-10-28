# %%
import numpy as np
import cv2
import os, argparse
import logging
from math import sqrt, pi, sin, cos, tan

class MultiCamTriang(object):
    """docstring for MultiCamTriang"""
    def __init__(self, poses, eye, Ks):
        super(MultiCamTriang, self).__init__()
        self.poses = poses                   # shape:(num_cam, c2w(3, 3)) transform matrix from ccs to wcs
        self.eye = eye                       # shape:(num_cam, 1, xyz(3)) camera position in wcs
        self.Ks = Ks                         # shape:(num_cam, K(3,3)) intrinsic matrix
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2 # shape:(num_cam) focal length
        self.p = Ks[:,0:2,2]                 # shape:(num_cam, xy(2)) principal point
        # self.projection_mat = projection_mat

    def setTrack2Ds(self, newtrack2Ds):     # must set, then run calculate3D
        #logging.debug('setting track2Ds:{newtrack2Ds}')
        self.track2Ds = newtrack2Ds          # shape:(num_cam, num_frame, xy(2)) 2D track from TrackNetV2

    def setCamInfo(self, cam_fid_pairs):
        # cam_fid_pairs: list of tuples (cam_id, fid)
        self.cam_fid_pairs = cam_fid_pairs    # shape:(num_cam, num_frame) camera id and frame id pairs
        self.num_cam = len(cam_fid_pairs)     # number of cameras
        self.num_frame = len(cam_fid_pairs[0]) if cam_fid_pairs else 0
    
    def calculate3D(self):
        self.num_cam, self.num_frame, _ = self.track2Ds.shape
        self.backProject()
        self.getApprox3D()
        return self.track3D

    def setProjectionMats(self, ProjectionMats):
        self.projection_mat = ProjectionMats

    def setPoses(self, poses):
        self.poses = poses

    def setEye(self, eye):
        self.eye = eye

    def setKs(self, ks):
        self.Ks = ks
        self.f = (ks[:,0,0] + ks[:,1,1]) / 2
        self.p = ks[:,0:2,2]

    def rain_calculate3D(self):
        total_track3Ds = np.zeros((1,3))
        track_3Ds = []
        for i in range(0,len(self.track2Ds)-1):
            for j in range(i+1,len(self.track2Ds)):
                self.track3D_homo = cv2.triangulatePoints(self.projection_mat[i],self.projection_mat[j],self.track2Ds[i][0],self.track2Ds[j][0]) # shape:(4,num_frame), num_frame=1
                self.track3D = self.track3D_homo[:3] / self.track3D_homo[3] # shape:(3,num_frame), num_frame=1
                self.track3D = np.stack(self.track3D, axis=1) # shape:(num_frame,3), num_frame=1
                track_3Ds.append(self.track3D)
                # logging.debug('i:{}, j:{}, self.track3D:{}'.format(i, j, self.track3D))
        track_3Ds = np.array(track_3Ds)
        n=1.5
        #IQR = Q3-Q1
        for i in range(0,3):
            IQR = np.percentile(track_3Ds[:,:,i],75) - np.percentile(track_3Ds[:,:,i],25)
            track_3Ds = track_3Ds[track_3Ds[:,:,i] <= np.percentile(track_3Ds[:,:,i],75)+n*IQR]
            track_3Ds = track_3Ds[:,np.newaxis,:]
        for i in track_3Ds:
            total_track3Ds += i
        return total_track3Ds/len(track_3Ds)
        # self.track3D_homo = cv2.triangulatePoints(self.projection_mat[0],self.projection_mat[1],self.track2Ds[0][0],self.track2Ds[1][0]) # shape:(4,num_frame), num_frame=1
        # self.track3D = self.track3D_homo[:3] / self.track3D_homo[3] # shape:(3,num_frame), num_frame=1
        # self.track3D = np.stack(self.track3D, axis=1) # shape:(num_frame,3), num_frame=1
        # return self.track3D
    
    def rain_calculate3D_plain(self):
        """
        Pairwise triangulation -> mean of all pairs (no IQR).
        """
        track_3Ds = []
        for i in range(0, len(self.track2Ds) - 1):
            for j in range(i + 1, len(self.track2Ds)):
                self.track3D_homo = cv2.triangulatePoints(self.projection_mat[i],self.projection_mat[j],self.track2Ds[i][0],self.track2Ds[j][0]) # shape:(4,num_frame), num_frame=1
                self.track3D = self.track3D_homo[:3] / self.track3D_homo[3] # shape:(3,num_frame), num_frame=1
                self.track3D = np.stack(self.track3D, axis=1) # shape:(num_frame,3), num_frame=1
                track_3Ds.append(self.track3D)

        if not track_3Ds:
            return None

        track_3Ds = np.vstack(track_3Ds)       # k x 3
        mean_X = track_3Ds.mean(axis=0, keepdims=True)  # 1 x 3
        return mean_X

    def rain_calculate3D_robust(self,
                     REPROJ_THRESH=10.0,     # px：重投影閾值
                     MIN_INLIERS=1,          # 至少要有幾顆候選通過
                     CLUSTER_EPS=0.5,        # m：同群距離半徑（可調）
                     Z_FLOOR_TOL=0.1,
                     meta=False):      
        # logging.debug(f"[Triangulate] Start triangulation with {len(self.track2Ds)} cameras.")

        # ① pairwise 產生候選
        cand_meta = []  # list of dict: {'X':(3,), 'pair':(i,j), 'errs':list}
        for i in range(len(self.track2Ds)-1):
            for j in range(i+1, len(self.track2Ds)):
                X_h = cv2.triangulatePoints(self.projection_mat[i],
                                            self.projection_mat[j],
                                            self.track2Ds[i][0],
                                            self.track2Ds[j][0])  # 4×1
                X = (X_h[:3] / X_h[3]).flatten()                    # (3,)
                # logging.debug(f"[Candidate] pair=({i},{j})  cams=({self.cam_fid_pairs[i]}, {self.cam_fid_pairs[j]})  X={np.round(X,3)}")
                if X[2] < -Z_FLOOR_TOL:
                    # logging.debug(f"[Reject-Z] pair=({i},{j}) Z={X[2]:.3f} < -{Z_FLOOR_TOL:.3f} -> drop candidate")
                    continue

                # ② 重投影誤差（對所有有量測的相機）
                errs = [self._reproj_err_one_cam(X, c) for c in range(len(self.track2Ds))]
                passed = (np.array(errs) < REPROJ_THRESH).sum() >= (len(self.track2Ds)+1)//2
                # logging.debug(f"[Reproj]    pair=({i},{j}) errs={np.round(errs,2)}  {'OK' if passed else 'REJECT'}")
                if passed:
                    cand_meta.append({'X': X, 'pair': (i, j), 'errs': np.array(errs, dtype=float)})

        if len(cand_meta) < MIN_INLIERS:
            # logging.warning("[Triangulate] All candidates rejected at reprojection stage.")
            return None

        # ③ 3D 距離分群（只在 inliers 之間）
        Xs = np.vstack([m['X'] for m in cand_meta])        # M×3
        # 距離矩陣 D (M×M)
        diff = Xs[:, None, :] - Xs[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        # logging.debug(f"[Cluster] Distance matrix (m):\n{np.round(D,3)}")

        # 建立半徑鄰接圖（距離 <= CLUSTER_EPS 視為連通）
        adj = D <= CLUSTER_EPS
        np.fill_diagonal(adj, True)

        # 找連通成分（clusters）
        M = len(cand_meta)
        visited = np.zeros(M, dtype=bool)
        clusters = []
        for s in range(M):
            if visited[s]: 
                continue
            stack = [s]
            comp = []
            while stack:
                u = stack.pop()
                if visited[u]: 
                    continue
                visited[u] = True
                comp.append(u)
                nbrs = np.where(adj[u])[0].tolist()
                for v in nbrs:
                    if not visited[v]:
                        stack.append(v)
            clusters.append(comp)

        # ---- 列印每個 cluster 的詳細成員 ----
        for idx, comp in enumerate(clusters):
            Xc = Xs[comp]                                  # (K,3)
            centroid = Xc.mean(axis=0)                     # (3,)
            mean_err = np.mean([cand_meta[k]['errs'].mean() for k in comp])
            # logging.debug(f"[Cluster#{idx}] size={len(comp)} "
            #             f"centroid={np.round(centroid,3)} "
            #             f"mean_reproj_err={mean_err:.2f}px")

            for k in comp:
                pair = cand_meta[k]['pair']                # (i, j)
                cams = (self.cam_fid_pairs[pair[0]], self.cam_fid_pairs[pair[1]])
                X    = cand_meta[k]['X']
                errs = cand_meta[k]['errs']
                # logging.debug(f"  - cand#{k} pair={pair} cams={cams} "
                #             f"X={np.round(X,3)} errs={np.round(errs,2)}")
                dist2cent = np.linalg.norm(X - centroid)
                # logging.debug(f"      dist_to_centroid={dist2cent:.3f}")

        # # 以「最大群」為主；若同大小，取群內平均重投影誤差最小者
        # def cluster_score(comp):
        #     return np.mean([cand_meta[k]['errs'].mean() for k in comp])
        # sizes = [len(c) for c in clusters]
        # best_size = max(sizes)
        # best_idxs = [idx for idx, c in enumerate(clusters) if len(c) == best_size]
        # if len(best_idxs) > 1:
        #     best_idx = min(best_idxs, key=lambda idx: cluster_score(clusters[idx]))
        # else:
        #     best_idx = best_idxs[0]
        # best_cluster = clusters[best_idx]
        # logging.debug(f"[Cluster] eps={CLUSTER_EPS} m  -> {len(clusters)} clusters; "
        #             f"pick cluster#{best_idx} with size={len(best_cluster)}; "
        #             f"members={best_cluster}")

        # ---- 以 size → 支持度 → 中位數 → 75分位數 進行 robust 平手裁決 ----
        def _candidate_support_and_stats(errs, t_strict=3.0):
            """
            errs: np.ndarray(num_cams,)  該候選在各相機上的重投影誤差(px)
            t_strict: 嚴格門檻，多少 px 以內算“好”的視角
            return:
            support: 該候選有多少視角 e < t_strict
            med:     誤差中位數（對極大值不敏感）
            q75:     誤差 75% 分位數（衡量尾端大小）
            """
            errs = np.asarray(errs, dtype=float)
            support = int((errs < t_strict).sum())
            med = float(np.median(errs))
            q75 = float(np.quantile(errs, 0.75))
            return support, med, q75
        
        def cluster_key(comp, t_strict=3.0):
            # 聚合該群內的“支持度總和”，以及候選的中位數/75分位數的中位
            supports, meds, q75s = [], [], []
            for k in comp:
                s, med, q75 = _candidate_support_and_stats(cand_meta[k]['errs'], t_strict=t_strict)
                supports.append(s)
                meds.append(med)
                q75s.append(q75)
            support_sum = int(np.sum(supports))          # 群內總支持度（越大越好）
            med_of_meds = float(np.median(meds))         # 群內“候選中位數”的中位（越小越好）
            med_of_q75s = float(np.median(q75s))         # 群內“候選 75% 分位”的中位（越小越好）
            return (-support_sum, med_of_meds, med_of_q75s), (support_sum, med_of_meds, med_of_q75s)

        # 1) 以群大小最大優先
        sizes = [len(c) for c in clusters]
        best_size = max(sizes)
        cands_same_size = [idx for idx, c in enumerate(clusters) if len(c) == best_size]

        # 2) 若同大小，用 robust 指標做平手裁決
        if len(cands_same_size) > 1:
            ranked = []
            for idx in cands_same_size:
                key, stats = cluster_key(clusters[idx], t_strict=REPROJ_THRESH)  # 用同一門檻當嚴格標準
                ranked.append((key, idx, stats))
                # logging.debug(f"[Cluster] cluster#{idx}: support_sum={stats[0]}, median(err)={stats[1]:.3f}px, q75(err)={stats[2]:.3f}px")
            ranked.sort(key=lambda x: x[0])  # 依 (-support_sum, med_of_meds, med_of_q75s) 升序
            best_idx = ranked[0][1]
            # ss, mm, qq = ranked[0][2]
            # # logging.debug(f"[Cluster] tie -> pick by robust key: cluster#{best_idx}, support_sum={ss}, "
            # #             f"median(err)={mm:.2f}px, q75(err)={qq:.2f}px")
        else:
            best_idx = cands_same_size[0]

        best_cluster = clusters[best_idx]
        # logging.debug(f"[Cluster] eps={CLUSTER_EPS} m  -> {len(clusters)} clusters; "
        #             f"pick cluster#{best_idx} with size={len(best_cluster)}; "
        #             f"members={best_cluster}")


        # ④ 同群平均（或可改 weighted average）
        X_cluster = Xs[best_cluster]            # K×3
        final_X = X_cluster.mean(axis=0)        # (3,)
        # logging.debug(f"[Triangulate] Final 3D={np.round(final_X,3)}  "
        #             f"(avg of {len(best_cluster)} clustered inliers)")

        if meta:
            used_pairs = [cand_meta[k]['pair'] for k in best_cluster]   # [(i,j), ...]（local index）
            used_cams_local = sorted({p for ij in used_pairs for p in ij})

            meta = {
                "cand_count": len(cand_meta),
                "n_clusters": len(clusters),
                "picked": {"size": len(best_cluster)},
                "used_pairs_local": used_pairs,
                "used_cams_local": used_cams_local,
            }
            if hasattr(self, "cam_fid_pairs"):
                try:
                    used_cams_global = sorted({ self.cam_fid_pairs[idx][0] for idx in used_cams_local })
                    meta["used_cams_global"] = used_cams_global
                    meta["used_cam_fids"] = {
                        int(self.cam_fid_pairs[idx][0]): int(self.cam_fid_pairs[idx][1])
                        for idx in used_cams_local
                    }
                except Exception:
                    pass

            return final_X.reshape(1, 3), meta
        
        return final_X.reshape(1, 3)
    

    def rain_calculate3D_by_method(self,
                               REPROJ_THRESH=10.0,     # px：重投影閾值
                               MIN_INLIERS=1,          # 至少要有幾顆候選通過（用於 reproj_only / reproj_cluster）
                               CLUSTER_EPS=0.5,        # m：同群距離半徑
                               Z_FLOOR_TOL=0.1,
                               meta=False,
                               method="reproj_cluster"):  # "reproj_only" | "cluster_only" | "reproj_cluster"
        """
        method:
            - "reproj_only":     只做重投影篩選，直接平均
            - "cluster_only":    不做重投影篩選（保留地板濾除），只做 3D 分群
            - "reproj_cluster":  先重投影篩選再 3D 分群（原 robust）
        """
        n_cam = len(self.track2Ds)
        logging.debug(f"[Triangulate] Start triangulation with {n_cam} cameras. method={method}")

        # ---------- ① pairwise 產生候選 ----------
        cand_meta_all = []  # 不帶重投影門檻濾除的「全部候選」（僅過 Z 濾除）
        has_cam_pairs = hasattr(self, "cam_fid_pairs")
        for i in range(n_cam-1):
            for j in range(i+1, n_cam):
                X_h = cv2.triangulatePoints(self.projection_mat[i],
                                            self.projection_mat[j],
                                            self.track2Ds[i][0],
                                            self.track2Ds[j][0])  # 4×1
                if X_h[3] == 0:
                    logging.debug(f"[Candidate] pair=({i},{j}) homogeneous w=0 -> skip")
                    continue
                X = (X_h[:3] / X_h[3]).flatten()                  # (3,)
                cams_dbg = (self.cam_fid_pairs[i], self.cam_fid_pairs[j]) if has_cam_pairs else ((i, None), (j, None))
                logging.debug(f"[Candidate] pair=({i},{j})  cams={cams_dbg}  X={np.round(X,3)}")
                if X[2] < -Z_FLOOR_TOL:
                    logging.debug(f"[Reject-Z] pair=({i},{j}) Z={X[2]:.3f} < -{Z_FLOOR_TOL:.3f} -> drop candidate")
                    continue

                # 重投影誤差（為了 meta 與後續可能使用；cluster_only 不以此門檻篩掉）
                errs = [self._reproj_err_one_cam(X, c) for c in range(n_cam)]
                cand_meta_all.append({'X': X, 'pair': (i, j), 'errs': np.array(errs, dtype=float)})

        if len(cand_meta_all) == 0:
            logging.warning("[Triangulate] No valid pairwise candidates (after Z-floor filtering).")
            return None

        # ---------- ② 依 method 決定是否做重投影門檻篩選 ----------
        if method in ("reproj_only", "reproj_cluster"):
            cand_meta = []
            half_cams_ok = np.array([
                (cm['errs'] < REPROJ_THRESH).sum() >= (n_cam + 1)//2
                for cm in cand_meta_all
            ], dtype=bool)
            for ok, cm in zip(half_cams_ok, cand_meta_all):
                logging.debug(f"[Reproj] pair={cm['pair']} errs={np.round(cm['errs'],2)}  {'OK' if ok else 'REJECT'}")
                if ok:
                    cand_meta.append(cm)

            if len(cand_meta) < MIN_INLIERS:
                logging.warning("[Triangulate] Candidates rejected at reprojection stage.")
                return None
        else:
            # cluster_only：直接使用 cand_meta_all（不以 REPROJ_THRESH 濾除）
            cand_meta = cand_meta_all

        # ---------- ③（選配）3D 距離分群 ----------
        def pick_by_clustering(cmeta):
            Xs = np.vstack([m['X'] for m in cmeta])  # M×3
            if len(cmeta) == 1:
                # 單一候選：視為一個群
                return Xs[0], [0], [cmeta[0]['pair']], sorted({p for p in cmeta[0]['pair']}), 1

            diff = Xs[:, None, :] - Xs[None, :, :]
            D = np.linalg.norm(diff, axis=2)
            adj = D <= CLUSTER_EPS
            np.fill_diagonal(adj, True)

            # 連通成分
            M = len(cmeta)
            visited = np.zeros(M, dtype=bool)
            clusters = []
            for s in range(M):
                if visited[s]:
                    continue
                stack = [s]
                comp = []
                while stack:
                    u = stack.pop()
                    if visited[u]:
                        continue
                    visited[u] = True
                    comp.append(u)
                    nbrs = np.where(adj[u])[0].tolist()
                    for v in nbrs:
                        if not visited[v]:
                            stack.append(v)
                clusters.append(comp)

            # 列印細節
            for idx, comp in enumerate(clusters):
                Xc = Xs[comp]
                centroid = Xc.mean(axis=0)
                mean_err = np.mean([cmeta[k]['errs'].mean() for k in comp])
                logging.debug(f"[Cluster#{idx}] size={len(comp)} centroid={np.round(centroid,3)} mean_reproj_err={mean_err:.2f}px")
                for k in comp:
                    pair = cmeta[k]['pair']
                    cams = (self.cam_fid_pairs[pair[0]], self.cam_fid_pairs[pair[1]]) if has_cam_pairs else ((pair[0],None),(pair[1],None))
                    X    = cmeta[k]['X']
                    errs = cmeta[k]['errs']
                    logging.debug(f"  - cand#{k} pair={pair} cams={cams} X={np.round(X,3)} errs={np.round(errs,2)} "
                                f"dist_to_centroid={np.linalg.norm(X - centroid):.3f}")

            # robust 平手裁決：size → 支持度總和 → 中位數 → 75 分位
            def _candidate_support_and_stats(errs, t_strict=3.0):
                errs = np.asarray(errs, dtype=float)
                support = int((errs < t_strict).sum())
                med = float(np.median(errs))
                q75 = float(np.quantile(errs, 0.75))
                return support, med, q75

            def cluster_key(comp, t_strict=REPROJ_THRESH):
                supports, meds, q75s = [], [], []
                for k in comp:
                    s, med, q75 = _candidate_support_and_stats(cmeta[k]['errs'], t_strict=t_strict)
                    supports.append(s); meds.append(med); q75s.append(q75)
                support_sum = int(np.sum(supports))
                med_of_meds = float(np.median(meds))
                med_of_q75s = float(np.median(q75s))
                return (-support_sum, med_of_meds, med_of_q75s), (support_sum, med_of_meds, med_of_q75s)

            sizes = [len(c) for c in clusters]
            best_size = max(sizes)
            cands_same_size = [idx for idx, c in enumerate(clusters) if len(c) == best_size]

            if len(cands_same_size) > 1:
                ranked = []
                for idx in cands_same_size:
                    key, stats = cluster_key(clusters[idx])
                    ranked.append((key, idx, stats))
                    logging.debug(f"[Cluster] cluster#{idx}: support_sum={stats[0]}, median(err)={stats[1]:.3f}px, q75(err)={stats[2]:.3f}px")
                ranked.sort(key=lambda x: x[0])
                best_idx = ranked[0][1]
            else:
                best_idx = cands_same_size[0]

            best_cluster = clusters[best_idx]
            X_cluster = Xs[best_cluster]
            final_X = X_cluster.mean(axis=0)
            used_pairs_local = [cmeta[k]['pair'] for k in best_cluster]
            used_cams_local = sorted({p for ij in used_pairs_local for p in ij})
            return final_X, best_cluster, used_pairs_local, used_cams_local, len(clusters)

        # ---------- ④ 依 method 產生最終點 ----------
        if method == "reproj_only":
            Xs = np.vstack([m['X'] for m in cand_meta])  # 直接對通過重投影門檻者平均
            final_X = Xs.mean(axis=0)
            used_pairs_local = [m['pair'] for m in cand_meta]
            used_cams_local = sorted({p for ij in used_pairs_local for p in ij})
            n_clusters = 1
            picked_size = len(cand_meta)
            picked_members = list(range(len(cand_meta)))  # meta 裡就放 index 列表
        elif method == "cluster_only":
            final_X, best_cluster, used_pairs_local, used_cams_local, n_clusters = pick_by_clustering(cand_meta)
            picked_size = len(best_cluster)
            picked_members = best_cluster
        else:  # "reproj_cluster"
            final_X, best_cluster, used_pairs_local, used_cams_local, n_clusters = pick_by_clustering(cand_meta)
            picked_size = len(best_cluster)
            picked_members = best_cluster

        final_X = final_X.reshape(1, 3)

        # ---------- ⑤ 回傳 ----------
        if not meta:
            return final_X

        # cand_count（給下游相容用）：對於有做重投影的模式 → 篩選後的數量；否則用全部
        cand_count_after = len(cand_meta) if method in ("reproj_only", "reproj_cluster") else len(cand_meta_all)

        out = {
            "method": method,
            "cand_count": int(cand_count_after),         # 相容欄位
            "cand_count_all": int(len(cand_meta_all)),   # 額外資訊
            "n_clusters": int(n_clusters) if n_clusters is not None else None,
            "picked": {"size": int(picked_size), "members": list(map(int, picked_members))},
            "used_pairs_local": [list(map(int, p)) for p in used_pairs_local],
            "used_cams_local": [int(x) for x in used_cams_local],
        }

        # 轉成 global（若有 cam_fid_pairs）
        if has_cam_pairs:
            try:
                cam_fid_pairs = list(self.cam_fid_pairs)  # [(global_cam_id, fid), ...] aligned to local index
                local2global = {loc: int(cam_fid_pairs[loc][0]) for loc in range(len(cam_fid_pairs))}
                # used_pairs_global
                upg = [[local2global[i], local2global[j]] for (i, j) in used_pairs_local if i in local2global and j in local2global]
                out["used_pairs_global"] = upg
                # used_cams_global
                out["used_cams_global"] = sorted({local2global[i] for i in used_cams_local if i in local2global})
                # used_cam_fids
                out["used_cam_fids"] = {int(cam_id): int(fid) for (cam_id, fid) in cam_fid_pairs}
            except Exception:
                pass

        return final_X, out

    
    def _reproj_err_one_cam(self, X, cam_idx):
        """
        以像素為單位回傳 3D 點 X 在 cam_idx 上的重投影誤差
        """
        P = self.projection_mat[cam_idx]       # 3×4
        X_h = np.append(X, 1.0)               # homogeneous
        x_h = P @ X_h                         # (3,)
        if x_h[2] == 0:                       # 避免除 0
            return np.inf
        u_pred, v_pred = x_h[:2] / x_h[2]
        u_meas, v_meas = self.track2Ds[cam_idx][0]
        return np.linalg.norm([u_pred - u_meas, v_pred - v_meas])
    
    def backProject(self):
        # Back project the 2d points of all frames to the 3d ray in world coordinate system

        # Shift origin to principal point
        self.track2Ds_ccs = self.track2Ds - self.p[:,None,:]

        # Back project 2D track to the CCS
        self.track2Ds_ccs = self.track2Ds_ccs / self.f[:,None,None]
        track_d = np.ones((self.num_cam, self.num_frame, 1))
        self.track2Ds_ccs = np.concatenate((self.track2Ds_ccs, track_d), axis=2)

        # 2D track described in WCS
        self.track2D_wcs = self.poses @ np.transpose(self.track2Ds_ccs, (0,2,1)) # shape:(num_cam, 3, num_frame)
        self.track2D_wcs = np.transpose(self.track2D_wcs, (0,2,1)) # shape:(num_cam, num_frame, 3)
        self.track2D_wcs = self.track2D_wcs / np.linalg.norm(self.track2D_wcs, axis=2)[:,:,None]

    def getApprox3D(self):
        # Calculate the approximate solution of the ball postition by the least square method
        # n-lines intersection == 2n-planes intersection

        planeA = np.copy(self.track2D_wcs)
        planeA[:,:,0] = 0
        planeA[:,:,1] = -self.track2D_wcs[:,:,2]
        planeA[:,:,2] = self.track2D_wcs[:,:,1]

        # check norm == 0
        planeA_tmp = np.copy(self.track2D_wcs)
        planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,2]
        planeA_tmp[:,:,1] = 0
        planeA_tmp[:,:,2] = self.track2D_wcs[:,:,0]
        mask = np.linalg.norm(planeA, axis=2)==0
        planeA[mask] = planeA_tmp[mask]

        # # check norm == 0
        # planeA_tmp = np.copy(self.track2D_wcs)
        # planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,1]
        # planeA_tmp[:,:,1] = self.track2D_wcs[:,:,0]
        # planeA_tmp[:,:,2] = 0
        # mask = np.linalg.norm(planeA, axis=2)==0
        # planeA[mask] = planeA_tmp[mask]

        planeB = np.cross(self.track2D_wcs, planeA)

        Amtx = np.concatenate((planeA, planeB), axis=0) # shape:(2num_cam, num_frame, 3)
        b = np.concatenate((self.eye*planeA, self.eye*planeB), axis=0).sum(-1)[:,:,None] # shape:(2num_cam, num_frame, 1)

        Amtx = np.transpose(Amtx, (1,0,2)) # shape:(num_frame, 2num_cam, 3)
        b = np.transpose(b, (1,0,2)) # shape:(num_frame, 2num_cam, 1)

        left = np.transpose(Amtx, (0,2,1)) @ Amtx # shape:(num_frame, 3, 3)
        right = np.transpose(Amtx, (0,2,1)) @ b # shape:(num_frame, 3, 1)

        self.track3D = np.linalg.pinv(left) @ right # shape:(num_frame, 3, 1)
        self.track3D = self.track3D.reshape(-1,3)
        '''
        [[-1.52680479,  2.06202114,  2.04934252],
        [-1.34739384,  1.67499119,  2.49134506],
        [-1.16905542,  1.29068153,  2.88204759],
        [-0.9924781,   0.91104616,  3.22891526],
        [-0.81207975,  0.5231891,   3.53126069],
        [-0.63397124,  0.14283066,  3.78343654],
        [-0.46045112, -0.23837983,  3.99178557],
        [-0.28176591, -0.62200495,  4.15098372],
        [-0.10533186, -1.00410534,  4.26820641],
        [ 0.07375752, -1.38452603,  4.337159  ],
        [ 0.25253153, -1.77248105,  4.36025826],
        [ 0.43151949, -2.15530492,  4.33638467],
        [ 0.61271096, -2.53986812,  4.26825079],
        [ 0.78954231, -2.91919479,  4.14984721],
        [ 0.96996572, -3.30069187,  3.9908667 ],
        [ 1.14671596, -3.68684962,  3.78039425],
        [ 1.32658648, -4.07093151,  3.52737657],
        [ 1.5030584,  -4.45345251,  3.22569203],
        [ 1.68413826, -4.8356843,   2.88023936],
        [ 1.86211489, -5.22080128,  2.48178244],
        [ 2.0396313,  -5.60385936,  2.04277793]]
        '''
'''
    mct = MultiCamTriang(
        track2Ds=np.array(track2Ds), 
        poses=np.array(poses), 
        eye=np.array(eye), 
        Ks=np.array(Ks)
'''
