function [ap, mAP] = eval_classification

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);


% initialize VOC options
VOCinit;

% train and test classifier for each class
ap = zeros(1, VOCopts.nclasses);
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    
    [recall,prec,ap(i)]=VOCevalcls(VOCopts,'comp2',cls,false);   % compute and display PR
    
%     if i<VOCopts.nclasses
%         fprintf('press any key to continue with next class...\n');
%         drawnow;
%         pause;
%     end
end
mAP = mean(ap);
fprintf('the mAP is %g.\n', mAP);

end