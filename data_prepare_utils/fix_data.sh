!/usr/env bash
dir=$1
for f in $dir/*.lbl;
do
	echo "processing $f"
	cat $f | cut -d ' ' -f 1 > $f.tmp
	mv $f.tmp $f
	rm -f $f.tmp
done

