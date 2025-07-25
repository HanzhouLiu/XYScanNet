%p = genpath("G:\RESEARCH\PHD\Motion_Deblurred\xyscannet\visualization_bidir\comparison\epoch3k\uni");
%gt = genpath("G:\RESEARCH\PHD\Motion_Deblurred\xyscannet\visualization_bidir\comparison\epoch3k\gt");

length_p = size(p,2);
path = {};
temp = [];
for i = 1:length_p 
    if p(i) ~= ';'
        temp = [temp p(i)]; 
    else 
        temp = [temp '\'];
        path = [path ; temp];
        temp = [];
    end
end  
clear p length_p temp;
length_gt = size(gt,2);
path_gt = {};
temp_gt = [];
for i = 1:length_gt 
    if gt(i) ~= ';'
        temp_gt = [temp_gt gt(i)];
    else 
        temp_gt = [temp_gt '\']; 
        path_gt = [path_gt ; temp_gt];
        temp_gt = [];
    end
end  
clear gt length_gt temp_gt;

file_num = size(path,1);
total_niqe = 0;
n = 0;
for i = 1:file_num
    file_path =  path{i}; 
    gt_file_path = path_gt{i};
    img_path_list = dir(strcat(file_path,'*.png'));
    gt_path_list = dir(strcat(gt_file_path,'*.png'));
    img_num = length(img_path_list);
    if img_num > 0
        for j = 1:img_num
            image_name = img_path_list(j).name;
            gt_name = gt_path_list(j).name;
            image =  imread(strcat(file_path,image_name));
            gt = imread(strcat(gt_file_path,gt_name));
            size(image);
            size(gt);
            cur_niqe = niqe(image);
            fprintf('%d', cur_niqe);
            total_niqe = total_niqe + cur_niqe;
            n = n + 1
        end
    end
end
niqe_score = total_niqe / n
close all;clear all;

