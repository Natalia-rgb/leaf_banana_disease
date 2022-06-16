%Folder_name = 'D:\Belajar\KULIAH\SKRIPSI\dataset\Dataset';
outputFolder = fullfile('D:\Belajar\KULIAH\SKRIPSI\dataset\Dataset\TEST'); 
rootFolder = fullfile(outputFolder);
categories = {'COVID','Normal','Viral_Pneumonia'};
corelImageSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
%[testingSet,trainingSet] = splitEachLabel(corelImageSet, 0.2,.8, 'randomized');

Totaltrainimages = numel(corelImageSet.Files);

net=resnet50();
net.Layers;
layer='fc1000';

Train_Feature = [];
for CurrentImage=1:Totaltrainimages
    queryimage=readimage(corelImageSet,CurrentImage);
%     imshow(queryimage) 
%     I=imresize(queryimage,[227,227]);
%     if size(I,3) == 1
%     I = repmat(I,[1 1 3]);
%     end
%     net = alexnet ; 
%     net.Layers;
%     layer='fc7';
%     F1 = activations(net,I,layer,'outputAs','rows');
%     
%     I=imresize(queryimage, [224 224]);
%     if size(I,3) == 1
%     I = repmat(I,[1 1 3]);
%     end
%     net = resnet50();
%     featureLayer = 'fc1000';
%     F2 = activations(net, I, featureLayer,'OutputAs', 'rows');
% 
%     F3 = Letrist_Descriptor(queryimage);
    
        I=imresize(queryimage,[224,224]);
        if size(I,3) == 1
        I = repmat(I,[1 1 3]);
        end

        F4=activations(net,I,layer,'outputAs','rows');
    
    
    
    Train_Feature=[Train_Feature; [F4]];
    CurrentImage
end
%save('Train_Feature','Train_Feature')
Train_Feature_label =[];
TrainLabel = corelImageSet.Labels
%Train_Feature_label = [Train_Feature Label]; 


Feature = [Train_Feature];
Label = [TrainLabel];
csvwrite('test_feature_resnet.csv', Feature);
writematrix(Label,'test_label_resnet.csv');
%csvwrite('label_vgg16.csv', Label);
%save('Feature_Label','Feature_Label')
% SVMStruct = fitcsvm(Train_Feature,trainingSet.Labels,'KernelFunction','linear');
% Group = predict(SVMStruct,Test_Feature);
% plotconfusion(testingSet.Labels,Group)