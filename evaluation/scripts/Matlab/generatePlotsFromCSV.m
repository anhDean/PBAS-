COLUMNS = {'N', 'Recall', 'Specificity', 'FPR', 'FNR', 'PBC', 'Precision', 'FMeasure'};
SOTA ={'PAWCS', 'CDet', 'SuBSENSE', 'MBS', 'PBAS2012'};
% sota_file ='C:\Users\Dinh\Documents\GitHub\Master\Code\PBAS+\ConsoleApplication2\code\PBAS-\evaluation\init\state_of_the_art_csv.dat'
% filename ='C:\Users\Dinh\Documents\GitHub\Master\Code\PBAS+\ConsoleApplication2\code\PBAS-\evaluation\data\N_eval_csv.dat';
S = csvread(sota_file, 1, 0);
M = csvread(filename, 1,0);
x = M(:,1);
sota_plots = [];

plotStyle = {'g--h','r:s','c-.x', 'k-*', 'm^-'}; % add as many as you need
for i=2:length(COLUMNS)
    y = M(:,i);
    figure1 = figure;
    hold on
    current = plot(x,y,'-ob');
    for j=1:length(SOTA)

        sota_plots = [sota_plots plot(x,S(j,i-1)*ones(1, length(x)), plotStyle{j})];
    end
    legend([current sota_plots], ['PBAS+' SOTA]);
    title([COLUMNS{i} ' ' date]);
    xlabel(COLUMNS{1});
    saveas(figure1, ['../../data/', COLUMNS{1}, '_', COLUMNS{i},'_eval_plot'], 'png');
end

close all
quit;