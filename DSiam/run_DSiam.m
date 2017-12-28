function [results ] = run_DSiam(seq,res_path,bSaveImg,rootdir,netname,nettype)

%%
%  This code is a demo for DSiam, a online updated deep tracker for fast tracking
%  If you use this code,please cite:
%  Qing Guo, Wei Feng, Ce Zhou, Rui Huang, Liang Wan, Song Wang. 
%  Learning Dynamic Siamese Network for Visual Object Tracking. In ICCV 2017.
%  
%  Qing Guo, 2017.
%%
%===================================%
%  Path initial setting
%===================================%
addpath(genpath([rootdir,'/matconvnet/matlab/']));
vl_setupnn ;
addpath([rootdir,'utils']);
addpath([rootdir,'models']);

%===================================%
%  Data initialization
%===================================%

s_frames = seq.s_frames;
initrect = seq.init_rect;

% read img
imgs = vl_imreadjpeg(s_frames,'numThreads', 12);
img  = imgs{1};
if size(img,3)==1
    img = repmat(img,1,1,3);
end

nFrames = numel(s_frames);

state = obj_initialize(img, initrect);
state.seq.frame = 1;
state.seq.nFrames = nFrames;
state.seq.seqname = seq.name;
state.seq.time = 0;
state.seq.isfail = 0;
state.seq.scores = zeros(nFrames);
state.seq.frame = 1;    

state.seq.scores(1) = 10;

% default
if nargin<4
    netname = 'vgg19';
    nettype = '1res';
end
[state, opts]...
    = fcnet_init(img, state,netname,nettype);

res=[initrect];            
duration = 0;

for it = 2:nFrames

    state.seq.frame = it;
%     fprintf('Processing frame %d/%d\n... ', state.seq.frame, nFrames);
    % **********************************
    % VOT: Get next frame
    % **********************************
    img = imgs{it};
    if size(img,3)==1
        img = repmat(img,1,1,3);
    end
    
    state = fcn_update(state, img,opts);
    initstate = [state.obj.targetLoc];
    
    duration =  state.seq.time;
    res = [res; initstate];
    
    %==================================
    %Display result
    %==================================
    if bSaveImg
        %-------------------------------Show the tracking results
        imshow(uint8(img));
        rectangle('Position',initstate,'LineWidth',4,'EdgeColor','r');
        hold on;
        text(5, 18, strcat('#',num2str(it)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
        set(gca,'position',[0 0 1 1]);
        pause(0.01);
        hold off;
        %         saveas(gcf,[res_path num2str(i) '.jpg'])
        %         imwrite(frame2im(getframe(gcf)),[res_path num2str(it) '.jpg']);
    end
    
end

results.res=res;
results.type='rect';
results.fps=(seq.len)/duration;
disp(['fps: ' num2str(results.fps)])
end

function [state] = obj_initialize(I, region, varargin)

gray = double(I(:,:,1));

[height, width] = size(gray);

% If the provided region is a polygon ...
if numel(region) > 4
    x1 = round(min(region(1:2:end)));
    x2 = round(max(region(1:2:end)));
    y1 = round(min(region(2:2:end)));
    y2 = round(max(region(2:2:end)));
    region = round([x1, y1, x2 - x1, y2 - y1]);
else
    region = round([round(region(1)), round(region(2)), ...
        round(region(1) + region(3)) - round(region(1)), ...
        round(region(2) + region(4)) - round(region(2))]);
end;

x1 = max(0, region(1));
y1 = max(0, region(2));
x2 = min(width-1, region(1) + region(3) - 1);
y2 = min(height-1, region(2) + region(4) - 1);

state.obj.pos = [y1 + y2 + 1, x1 + x2 + 1] / 2;
state.obj.targetsz = [y2-y1+1,x2-x1+1];
state.obj.base_targetsz = [y2-y1+1,x2-x1+1];
state.obj.targetLoc = [x1, y1, state.obj.targetsz([2,1])];
state.obj.change_alphaf = [];
state.obj.change_featf = [];

end

function state= fcn_update(state,img,opts)

%load state
targetsz = state.obj.targetsz.*opts.targetszrate;
pos = state.obj.pos;
s_x = state.obj.s_x;

corrfeat = state.obj.corrfeat;
isfail = state.seq.isfail;

net_conv = state.net.net_conv;
net_obj = state.net.net_obj;


%load params
instanceSize = opts.instanceSize;
window = opts.window;
min_s_x = opts.min_s_x;
max_s_x = opts.max_s_x;
avgChans = opts.avgChans;
scales = opts.scales;

scaledInstance = s_x .* scales;
scaledTarget = [targetsz(1) .* scales; targetsz(2) .* scales];

tic;

% extract scaled crops for search region x at previous target position
x_crops = make_scale_pyramid(img, pos, scaledInstance, instanceSize, avgChans,opts);%, opts,saliency_map);

% evaluate the offline-trained network for exemplar x features
[newTargetPosition, newScale, score,responseMap,scorePos] = tracker_eval(net_conv,round(s_x), ...
        corrfeat, x_crops, pos, window, opts);

pos = gather(newTargetPosition);

if opts.isupdate
%     score
    if score >0

        wc_z = targetsz(2) + opts.contextAmount*sum(targetsz);
        hc_z = targetsz(1) + opts.contextAmount*sum(targetsz);
        s_z = sqrt(wc_z*hc_z);
        [z_crop, ~] = get_subwindow_tracking(img, pos, ...
            [opts.exemplarSize opts.exemplarSize], [round(s_z) round(s_z)], opts.avgChans,opts.averageImage);
        z_crop = gpuArray(single(z_crop));
        
        net_obj.eval({opts.netobj_input, z_crop});
        state.seq.scores(state.seq.frame) = score;
        
        tcorrfeat{1} =  net_obj.vars(opts.obj_feat_id(1)).value;
        tcorrfeat{2} =  net_obj.vars(opts.obj_feat_id(2)).value; 
        
        % updating the target variation transformation
        if opts.vartransform
            net_conv = update_v(net_conv,corrfeat,tcorrfeat,opts);
        end
        
        % updating the background suppression transformation
        if opts.backsupression
            [x_back(:,:,:,1), ~] = get_subwindow_tracking(gather(img), pos,...
                [instanceSize instanceSize], [round(scaledInstance(newScale)) round(scaledInstance(newScale))], avgChans);
            x_back(:,:,:,2) = x_back(:,:,:,1).* opts.saliency_window;%state.obj.x_crop;%
            net_obj.eval({opts.netobj_input, gpuArray(x_back)});
            tcorrfeat{1} =  net_obj.vars(opts.obj_feat_id(1)).value;
            tcorrfeat{2} =  net_obj.vars(opts.obj_feat_id(2)).value;
            net_conv = update_w(net_conv,tcorrfeat,opts);
        end

        % scale damping and saturation
        if isfail
            wc_z = targetsz(2) + opts.contextAmount*sum(targetsz);
            hc_z = targetsz(1) + opts.contextAmount*sum(targetsz);
            s_z = sqrt(wc_z*hc_z);
            scale_z = opts.exemplarSize / s_z;
            d_search = (opts.instanceSize - opts.exemplarSize)/2;
            pad = d_search/scale_z;
            s_x = s_z + 2*pad;
            isfail = 0;
        else
            s_x = max(min_s_x, min(max_s_x, (1-opts.scaleLR)*s_x + opts.scaleLR*scaledInstance(newScale)));
            targetsz = (1-opts.scaleLR)*targetsz + opts.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
        end
        
    else
        isfail = 1;
        s_x = max(min_s_x, min(max_s_x, s_x*1.1));
        net_conv.layers(net_conv.getLayerIndex('circonv1_1')).block.enable = false;
        net_conv.layers(net_conv.getLayerIndex('circonv1_2')).block.enable = false;
        if strcmp(opts.nettype,'1res')
            net_conv.layers(net_conv.getLayerIndex('circonv2_1')).block.enable = false;
            net_conv.layers(net_conv.getLayerIndex('circonv2_2')).block.enable = false;
        end
    end
    
else
    % scale damping and saturation
    s_x = max(min_s_x, min(max_s_x, (1-opts.scaleLR)*s_x + opts.scaleLR*scaledInstance(newScale)));
    targetsz = (1-opts.scaleLR)*targetsz + opts.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
end

% validate 
tmp = pos+targetsz./2;
if tmp(1)<0||tmp(2)<0||tmp(1)>opts.imgsz(1)||tmp(2)>opts.imgsz(2)
   state.obj.failframes = state.obj.failframes+1;
   if state.obj.failframes>=2
      pos = [size(img,1),size(img,2)]./2;
      state.obj.failframes =0;
   end
   isfail = 1;
   net_conv.layers(net_conv.getLayerIndex('circonv1_1')).block.enable = false;
   net_conv.layers(net_conv.getLayerIndex('circonv1_2')).block.enable = false;
   if strcmp(opts.nettype,'1res')
       net_conv.layers(net_conv.getLayerIndex('circonv2_1')).block.enable = false;
       net_conv.layers(net_conv.getLayerIndex('circonv2_2')).block.enable = false;
   end
end

targetsz = targetsz./opts.targetszrate;
state.obj.s_x = s_x;
state.obj.pos = pos;
state.obj.targetsz = targetsz;
state.obj.targetLoc = [pos([2,1]) - targetsz([2,1])/2, targetsz([2,1])];

state.seq.time = state.seq.time + toc;
state.seq.isfail = isfail;

end

function [change_alpahf,change_featf] = update_change(corrfeat,new_corrfeat,lambda,issum)
if nargin<4
   issum =false; 
end

% leanring filter from corrfeat to new_corrfeat
cos_window = hann(size(corrfeat,1)) * hann(size(corrfeat,2))';
tcorrfeat = bsxfun(@times, corrfeat, cos_window);

corrfeatf = fft2(tcorrfeat);
numcorr = numel(corrfeatf(:,:,1));
if ~issum
    kcorrfeatf = (corrfeatf .* conj(corrfeatf))./numcorr;
else
    kcorrfeatf = sum(corrfeatf .* conj(corrfeatf),3)./numel(corrfeatf);
end
tnew_corrfeat = bsxfun(@times, new_corrfeat, cos_window);
tnew_corrfeatf = fft2(tnew_corrfeat);
alphaf = tnew_corrfeatf./ (kcorrfeatf+ lambda);   

change_alpahf = alphaf;
change_featf = corrfeatf;
end

% fast online learning for V
function net = update_v(net,feats_1,feats_t,p)

[alphaf,featf] = update_change(feats_1{1}(:,:,:,1),feats_t{1},p.v_lambda);

net.params(net.getParamIndex('cir11_alphaf')).value = alphaf;
net.params(net.getParamIndex('cir11_featf')).value = featf;
net.layers(net.getLayerIndex('circonv1_1')).block.enable = true;

if strcmp(p.nettype,'1res')
    [alphaf,featf] = update_change(feats_1{2}(:,:,:,1),feats_t{2},p.v1_lambda);    
    net.params(net.getParamIndex('cir21_alphaf')).value = alphaf;
    net.params(net.getParamIndex('cir21_featf')).value = featf;
    net.layers(net.getLayerIndex('circonv2_1')).block.enable = true;
end

end

% fast online learning for W
function net = update_w(net,feats,p)

[alphaf,featf] = update_change(feats{1}(:,:,:,1),feats{1}(:,:,:,2),p.w_lambda);
net.params(net.getParamIndex('cir12_alphaf')).value = alphaf;
net.params(net.getParamIndex('cir12_featf')).value = featf;
net.layers(net.getLayerIndex('circonv1_2')).block.enable = true;

if strcmp(p.nettype,'1res')
    [alphaf,featf] = update_change(feats{2}(:,:,:,1),feats{2}(:,:,:,2),p.w1_lambda);
    net.params(net.getParamIndex('cir22_alphaf')).value = alphaf;
    net.params(net.getParamIndex('cir22_featf')).value = featf;
    net.layers(net.getLayerIndex('circonv2_2')).block.enable = true;
end

end

