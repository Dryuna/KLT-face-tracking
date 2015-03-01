clear all;close all;

%%%%%%%%%%%%%%%%% Change data path %%%%%%%%%%%%%%%%%%%
data_path = '/your/path/to/data/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subjectID = 5; %Subject identification number
start_time = 200; %in seconds
time_range = 15; %in seconds

frontVidObj = VideoReader(strcat(data_path,num2str(subjectID),'/',num2str(subjectID),'.camera.mp4'));
nFrames = frontVidObj.NumberOfFrames;
frameRate = frontVidObj.FrameRate;

VideoFile='/path/to/saved/video';
writerObj = VideoWriter(VideoFile);
fps= 10; 
writerObj.FrameRate = fps;
open(writerObj);

%Detect face in frames
faceDetector = vision.CascadeObjectDetector;

tracking = false;

for i = round(start_time*frameRate) : round(start_time*frameRate + time_range*frameRate)
    if i <= nFrames || i >= 0
        frame = read(frontVidObj, i);  
        if tracking == true 
            % Track the points. Note that some points may be lost.
            [points, isFound] = step(pointTracker, frame);
            visiblePoints = points(isFound, :);
            oldInliers = oldPoints(isFound, :);

            if size(visiblePoints, 1) >= 2 % need at least 2 points

                % Estimate the geometric transformation between the old points
                % and the new points and eliminate outliers
                [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                    oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

                % Apply the transformation to the bounding box
                [bboxPolygon(1:2:end), bboxPolygon(2:2:end)] ...
                    = transformPointsForward(xform, bboxPolygon(1:2:end), bboxPolygon(2:2:end));

                % Insert a bounding box around the object being tracked
                frame = insertShape(frame, 'Polygon', bboxPolygon,'Color','Green','Opacity',1);
                
                % Display tracked points
                %frame = insertMarker(frame, visiblePoints, '+','Color', 'white');

                % Reset the points
                oldPoints = visiblePoints;
                setPoints(pointTracker, oldPoints);
            end
            
            %Do face detection just to print the output
            BB = step(faceDetector,frame); 
            for j = 1:size(BB,1)
                frame = insertShape(frame, 'Rectangle', BB(j,:),'Color','Red','Opacity',1);
            end
        end
        if tracking == false
            %Returns Bounding Box values based on number of objects
            BB = step(faceDetector,frame); 
            if size(BB,1) > 0 
                if size(BB,1) ~=1 
                    % Get the bigger face
                    [val idx] = max(BB(:,3));
                    BB = BB(idx,:);
                end   
                % Convert the first box to a polygon.
                % This is needed to be able to visualize the rotation of the object.
                x = BB(1, 1); y = BB(1, 2); w = BB(1, 3); h = BB(1, 4);
                bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];
            
                % Draw the returned bounding box around the detected face.
                frame = insertShape(frame, 'Polygon', bboxPolygon,'Color','Red');
            
                % Detect feature points in the face region.
                points = detectMinEigenFeatures(rgb2gray(frame), 'ROI', BB);

                % Display the detected points.
                hold on
                plot(points);

                % Create a point tracker and enable the bidirectional error constraint to
                % make it more robust in the presence of noise and clutter.
                pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

                % Initialize the tracker with the initial point locations and the initial
                % video frame.
                points = points.Location;
                initialize(pointTracker, points, frame);

                % Make a copy of the points to be used for computing the geometric
                % transformation between the points in the previous and the current frames
                oldPoints = points;
                
                tracking = true;    
            end
        end   
        
        imshow(frame); title('Detected face');
        writeVideo(writerObj,im2frame(frame));
    end
    
end


% Clean up
release(pointTracker);
close(writerObj);