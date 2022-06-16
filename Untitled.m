x_train = readtable('resnet50_X_train.csv', 'HeaderLines', 1);
x_train(:,1) = [];
x_train = table2array(x_train);
x_test = readtable('resnet50_X_test.csv', 'HeaderLines', 1);
x_test(:,1) = [];
y_train = readtable('resnet50_y_train.csv', 'HeaderLines', 1);
y_train(:,1) = [];
y_test = readtable('resnet50_y_test.csv', 'HeaderLines', 1);
y_test(:,1) = [];
x_test = table2array(x_test);



y_test = table2cell(y_test);
y_train = table2cell(y_train);
temp = templateSVM('BoxConstraint',12.989521109573113,'KernelFunction','gaussian','KernelScale',1/sqrt(0.017889493724961073))
Mdl = fitcecoc(x_train,y_train,'Coding','onevsall','learners',temp)
saveLearnerForCoder(Mdl,'SVMresnet50GA');
predictedLabels = predict(Mdl, x_test);
confusionmatrix = confusionmat(y_test,predictedLabels)
akurasi = (confusionmatrix(1,1)+confusionmatrix(2,2)+confusionmatrix(3,3))/778


        gambarp = imread("D:\Belajar\KULIAH\SKRIPSI\dataset\test\images\COVID-00002.jpg");
        I=imresize(gambarp,[224,224]);
        if size(I,3) == 1
        I = repmat(I,[1 1 3]);
        end
        net=resnet50();
        net.Layers;
        layer='fc1000';
        F4=activations(net,I,layer,'outputAs','rows');
        Train_Feature = [[F4]];
        nmfeature =mapminmax(Train_Feature,0,1);
        %nnmfeature =  (Train_Feature - min(Train_Feature)) / ( max(Train_Feature) - min(Train_Feature) );
        %nmfeature = nnmfeature * (1 - 0) + 0;
        Model = loadLearnerForCoder('SVMresnet50GA');
        label = predict(Model,nmfeature)
        
        conn = database('covid','localhost','');
        conn.Message
net = squeezenet;
analyzeNetwork(net)
        