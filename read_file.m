function [Sample, Sample_label, no_of_class,fp]=read_file()

%%
%............READ DATA...............

%PC
addpath(genpath('C:\Users\aniru\Desktop\ML\Multi-label\Mat data'));
addpath('C:\Users\aniru\Desktop\ML\Multi-label\Mat data');

%Laptop
% addpath(genpath('C:\Users\cssc\Documents\MATLAB\Program\Data\Mat data'));
% cd('C:\Users\cssc\Documents\MATLAB\Program\Data\Mat data');

%.........Normalize........
% 
%load('yeast_TTdata');
%no_of_class = 14;
%fp='yeast_TTdata';
 %load('emotions_TTdata');
 %no_of_class = 6;
%fp='emotions_TTdata';
 %load('mediamill_TTdata');
 %no_of_class = 101;
%fp='mediamill_TTdata';
% load('scene_TTdata');
 %no_of_class = 6;
%fp='scene_TTdata';
 %load('birds_TTdata');
 %no_of_class = 19;
%fp='birds_TTdata';
%load('bibtex_TTdata');
% no_of_class = 159;
%fp='bibtex_TTdata';
 %load('enron_TTdata');
 %no_of_class = 53;
%fp='enron_TTdata';
 %load('flags_data');
%no_of_class = 7;
%fp='flags_data';


%.......Dont normalized.......
 %load('Corel5k_TTdata');
 %no_of_class = 374;
%fp='Corel5k_TTdata';
 %load('Corel5k-sparse_TTdata');
 %no_of_class = 374;
%fp='Corel5k-sparse_TTdata';
%load('Arts1_TTdata');
%no_of_class = 26;
%fp='Arts1_TTdata';
% load('Business1_TTdata');
% no_of_class = 30;

% load('Computers1_TTdata');
% no_of_class = 33;

% load('Education1_TTdata');
% no_of_class = 33;

% load('Entertainment1_TTdata');
% no_of_class = 21;

% load('Health1_TTdata');
% no_of_class = 32;
%fp='Health1_TTdata';
% load('Recreation1_TTdata');
% no_of_class = 22;

% load('Reference1_TTdata');
% no_of_class = 33;

% load('Science1_TTdata');
% no_of_class = 40;

 %load('Social1_TTdata');
% no_of_class = 39;
%fp='Social1_TTdata';
% load('Society1_TTdata');
% no_of_class = 27;

%load('slashdot_data');
 %no_of_class = 22;
 %fp='slashdot_data';
% % 
 %load('llog_data');
 %no_of_class = 75;
 %fp='llog_data';
%fp='llog_data';
 %load('genbase_data');
 %fp='genbase_data';
 %no_of_class = 27;
% % 
%load('medical_data');
 %no_of_class = 45;
%fp='medical_data';
%All unique label sets
% load('CAL500_data');
 %no_of_class = 174;
%fp='CAL500_data';
%%
%PC
cd 'C:\Users\aniru\Desktop\ML\algo\ML-RBF(OWN)';

%Laptop
% cd 'C:\Users\cssc\Documents\MATLAB\Program\clustering';


%............................PREPARATION of DATA...........................
%Concat Train test
Sample = cat(1,train_data,test_data);

[no_of_sample, column] = size(Sample);
% Idx=randperm(no_of_sample);
% Sample1=Sample(Idx,:);

%genbase
% Sample=Sample(:,2:column);
% column=column-1;

no_of_feature = column - no_of_class;

Sample_label = zeros(no_of_sample, no_of_class, 'double');
Sample_label(:, 1:no_of_class) = Sample(:, (no_of_feature+1):column);

%Normalize
%maxval = max(Sample);
%minval = min(Sample);
%for j = 1:no_of_feature
 %      Sample(:,j) = (Sample(:,j) - minval(j))/(maxval(j)-minval(j));
%end
end