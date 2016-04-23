% somenotice about this code, right now use the min border. Some of the
% landmarks may not be include in the box



function GenerateData(target_size, img_dir, target_dir)
%  rng('default');
if (nargin==2)
    target_dir = [img_dir,'_',num2str(target_size)];
end
img_list = dir(['../',img_dir,'/*.jpg']);
for i = 1:length(img_list)
% j = round(rand()*length(img_list));
% j = 59;
% for i = j:j
    fprintf('\r%.2f%% reading %s',i/length(img_list)*100, ['../',img_dir,'/',img_list(i).name]);
    img = imread(['../',img_dir,'/',img_list(i).name]);
    dotloc = find(img_list(i).name == '.');
    pts = readPoints(['../',img_dir,'/',img_list(i).name(1:dotloc),'pts']);
    box = boundBox(pts);
    [h,w,~] = size(img);
    img = im2double(img);
%     box = [round(max(1,box(1))), round(min(w,box(2))),round(max(1,box(3))),round(min(h,box(4)))]
    s = round(max(box(2)-box(1), box(4)-box(3)));
% s = round(min(box(2)-box(1), box(4)-box(3)));
    s = min([s,w-1,h-1]);
%     cent = round([bo(1)+])
%     box = ((box(1)+box(2))/2-s,(box(1)+box(2))/2+s,(box(3)+box(4))/2-s,(box(3)+box(4))/2+s);
    origin = max([(box(1)+box(2))/2-s/2, (box(3)+box(4))/2-s/2; 1, 1],[],1);
%     origin = [(box(1)+box(2))/2-s/2, (box(3)+box(4))/2-s/2; 1, 1]
    
    origin = round(origin);
    if (origin(1)+s>w)
        origin(1) = w-s;
    end
    if (origin(2)+s>h)
        origin(2) = h-s;
    end
    box = [origin(1),origin(1)+s,origin(2), origin(2)+s];
    
    
    
%     visualize(img, pts, box)
    
%     imshow(img(box(3):box(4),box(1):box(2),:))
%     result_img = downsampling(img(box(3):box(4),box(1):box(2),:), target_size);
result_img = downsampling(img, target_size,box);
[rh,rw,~] = size(result_img);
    result_pts = translate(pts,origin,s/target_size);
%     visualize(result_img, result_pts, translate(box,origin,s/target_size))
%     figure()
%     imshow(result_img)
    if (~isequal(exist(['../',target_dir,'/'], 'dir'),7))
        mkdir(['../',target_dir,'/'])
    end
    imwrite(result_img, ['../',target_dir,'/',img_list(i).name(1:dotloc-1),'_',num2str(target_size),'.jpg']);
    writePoints(result_pts, ['../',target_dir,'/',img_list(i).name(1:dotloc-1),'_',num2str(target_size),'.pts']);
end


end

function marks = readPoints(pts_dir)
    i = 0;
    fh = fopen(pts_dir, 'r');
    if (fh == -1)
        fprintf('Failed to open the pts file');
    else
        marks = [];
        line = fgetl(fh);
        while (ischar(line) && ~isempty(line))
            [x, line] = strtok(line);
            [y, ~] = strtok(line);
            x = str2double(x);
            y = str2double(y);
            if (~isnan(x) && ~isnan(y))
                marks = [marks;[x,y,i]];
                i = i+1;
            end
            line = fgetl(fh);
        end
    end
    fclose(fh);
    
end

function writePoints(pts, filename)
fh = fopen(filename,'w');
for i =1:size(pts,1)
    fprintf(fh,'%f %f %f\n', pts(i,1), pts(i,2), pts(i,3));
end
fclose(fh);
end

function bound = boundBox(pts)
    left = min(pts(:,1));
    right = max(pts(:,1));
    up = min(pts(:,2));
    down = max(pts(:,2));
   
    wiggling = rand(1,4);
    wl = 0.05+0.2*wiggling(1)+1;
    wr = 0.05+0.2*wiggling(2)+1;
    wu = 0.05+0.2*wiggling(3)+1;
    wd = 0.05+0.1*wiggling(4)+1;
    midx = (left+right)/2;
    midy = (up+down)/2;
    bound = [midx+wl*(left-midx),midx+wr*(right-midx),midy+wu*(up-midy),midy+wd*(down-midy)];
    
end

function visualize(img, pts, box)
    h = figure();
    imshow(img);
    hold on
    plot(pts(:,1),pts(:,2),'rx');
    plot([box(1),box(1),box(2),box(2),box(1)],[box(3),box(4),box(4),box(3),box(3)],'g-');
    for i=1:length(pts(:,1))
        txt = num2str(i-1);
        text(pts(i,1),pts(i,2),txt)
    end
    hold off
end

function points = translate(pt,origin,scale)
    points = pt;
    points(:,1) = (pt(:,1)-origin(1))/scale;
    points(:,2) = (pt(:,2)-origin(2))/scale;
end

function dimg = downsampling(img, target_size,box)
     [ih,iw,~] = size(img);
h = box(2)-box(1)+1;
    rate = h/target_size;
    sigma = round(rate/3);
    filter = fspecial('gaussian',3*sigma,sigma);
    blured = imfilter(img, filter, 'symmetric');
    size(blured);
% blured = img;
%     sample_ind = round(linspace(1, h, target_size));
    samplehor = linspace(box(1), box(2), target_size);
    samplever = linspace(box(3), box(4), target_size);
    % sample the image with bilinear interpolation
    sh0 = floor(samplehor);
    sh1 = sh0+1;
    sh1(end) = min(sh1(end),iw);
%     sh1(end) = sh1(end)-1;
    
    sv0 = floor(samplever);
    sv1 = sv0+1;
    sv1(end) = min(sv1(end),ih);
%     sv1(end) = sv1(end)-1;
    
    
    
    lambdahor = samplehor - sh0;
    lambdaver = samplever - sv0;
    
    i00 = blured(sv0,sh0,:);
    i01 = blured(sv0,sh1,:);
    i10 = blured(sv1,sh0,:);
    i11 = blured(sv1,sh1,:);
    
    [hor,ver] = meshgrid(lambdahor,lambdaver);
    
    
    dimg = (1-hor).*(1-ver).*i00(:,:,1)+(1-ver).*hor.*i01(:,:,1)+ver.*hor.*i11(:,:,1)+ver.*(1-hor).*i10(:,:,1);
    dimg = cat(3,dimg,(1-hor).*(1-ver).*i00(:,:,2)+(1-ver).*hor.*i01(:,:,2)+ver.*hor.*i11(:,:,2)+ver.*(1-hor).*i10(:,:,2));
    dimg = cat(3,dimg,(1-hor).*(1-ver).*i00(:,:,3)+(1-ver).*hor.*i01(:,:,3)+ver.*hor.*i11(:,:,3)+ver.*(1-hor).*i10(:,:,3));
%     dimg = dimg(box(3):box(4),box(1):box(2),:);
%     dimg = blured(round(sample),round(sample),:);
    
end