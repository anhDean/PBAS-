COLUMNS = {'N', 'Recall', 'Specificity', 'FPR', 'FNR', 'PBC', 'Precision', 'FMeasure'};
M = csvread(filename, 1,0);
x = M(:,1);

for i=2:length(COLUMNS)
    y = M(:,i);
    figure1 = figure;
    plot(x,y);
    title([COLUMNS{i} ' ' date]);
    xlabel(COLUMNS{1});
    saveas(figure1, ['../../data/', COLUMNS{1}, '_', COLUMNS{i},'_eval_plot'], 'png');
end

close all
quit;