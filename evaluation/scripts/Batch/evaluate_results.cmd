rem 1: folder with evaluation script  2: dataset_root 3: output root
matlab -r "cd('%1'); processFolder('%~2', '%~3'); quit;"
exit