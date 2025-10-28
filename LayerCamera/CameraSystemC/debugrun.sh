#!/bin/bash

run () {
    GST_DEBUG=3 GST_DEBUG_DUMP_DOT_DIR=./gst_debug ./camera 2>&1 | tee debugrun.log
}

generate_png () {
    for filename in gst_debug/*.dot; do
        base=${filename%.dot}
        dot -Tpng "$filename" > "$base.png"
        # ... rest of the loop body
    done
}

if [ "$1" == "gen" ]; then
    generate_png
else
    mkdir gst_debug 2>/dev/null

    rm gst_debug/*.dot
    rm gst_debug/*.png

    run
    generate_png
fi