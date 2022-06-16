outputFolder = fullfile('D:\leaf-banana-disease\LeafBanana'); 
rootFolder = fullfile(outputFolder);
categories = {'sehat','sakit'};
corelImageSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
[testingSet,trainingSet] = splitEachLabel(corelImageSet, 0.3,.7, 'randomized');

Totaltrainimages = numel(trainingSet.Files);



Train_Feature = [];
for CurrentImage=1:Totaltrainimages
    
    queryimage=readimage(trainingSet,CurrentImage);
    Img = rgb2gray(queryimage);
    H = imhist(Img)';
    H = H/sum(H);
    I = [0:255];
    CiriMEAN = I*H';
    CiriENT = -H*log2(H+eps)';
    CiriVAR = (I-CiriMEAN).^2*H';
    CiriSKEW = (I-CiriMEAN).^3*H'/CiriVAR^1.5;
    CiriKURT = (I-CiriMEAN).^4*H'/CiriVAR^2-3;
    Train_Feature=[Train_Feature; [CiriMEAN,CiriENT,CiriVAR,CiriSKEW,CiriKURT]];
    CurrentImage
    
end
%save('Train_Feature','Train_Feature')
Train_Feature_label =[];
TrainLabel = [trainingSet.Labels]
%Train_Feature_label = [Train_Feature Label]; 

Totaltestingimages = numel(testingSet.Files);
Test_Feature  = [];


for CurrentImage=1:Totaltestingimages
   queryimage=readimage(testingSet,CurrentImage);
    Img = rgb2gray(queryimage);
    H = imhist(Img)';
    H = H/sum(H);
    I = [0:255];
    CiriMEAN = I*H';
    CiriENT = -H*log2(H+eps)';
    CiriVAR = (I-CiriMEAN).^2*H';
    CiriSKEW = (I-CiriMEAN).^3*H'/CiriVAR^1.5;
    CiriKURT = (I-CiriMEAN).^4*H'/CiriVAR^2-3;
   Test_Feature=[Test_Feature; [CiriMEAN,CiriENT,CiriVAR,CiriSKEW,CiriKURT]];
    CurrentImage
   
end

Test_Feature_label=[];
TestLabel = testingSet.Labels;
Feature = [Train_Feature;Test_Feature];
Label = [TrainLabel;TestLabel];
csvwrite('feature_squeezenet.csv', Feature);
writematrix(Label,'label_squeezenet.csv');
SVMStruct = fitcsvm(Train_Feature,string(trainingSet.Labels),'KernelFunction','linear');
saveLearnerForCoder(SVMStruct,'SVMleaf');
