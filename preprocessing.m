dataset = csvread('dataset.csv');
col = zeros(1151,1);
data = zeros(1151,20);
for i = 3:18
    col = dataset(:,i);
    for j=1:1151
        data(j,i) = (col(j)-min(col))/(max(col(j))-min(col(j)));
    end
end