import tensorflow as tf
import numpy as np
import cv2
import os
class yolov1:
    def __init__(self , S = 7 , B=2 , C=20, img_size=448):
        """
       Args:
            S: Grid size (S×S)
            B: Number of bounding boxes per grid cell
            C: Number of classes
            img_size: Input image size
        
        """
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size
        self.lambda_coord = 5.0 # from the paper 
        self.lambda_noobj = 0.5 # from the paper 
        self.model = self._build_model()
    
    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))

        #layer 1 
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
        
        #layer 2 
        x = tf.keras.layers.Conv2D(192, 3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
        
        #layer 3 
        x = tf.keras.layers.Conv2D(128, 1, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

        
        #layer 4 
        for _ in range(4):
            x = tf.keras.layers.Conv2D(256, 1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            x = tf.keras.layers.Conv2D(512, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            
        x = tf.keras.layers.Conv2D(512, 1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv2D(1024, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
        
        #layer 5
        for _ in range(2):
            x = tf.keras.layers.Conv2D(512, 1, strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            x = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
         
        #last layer 
        x = tf.keras.layers.Conv2D(1024, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = tf.keras.layers.Conv2D(1024, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = tf.keras.layers.Conv2D(1024, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        x = tf.keras.layers.Conv2D(1024, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)   
        
        #fc layer 
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
         # Output layer: S×S×(B*5+C)
        # For each grid cell, we predict B bounding boxes (each with 5 values: x, y, w, h, confidence)
        # and C class probabilities
        output_dim = self.S * self.S * (self.B * 5 + self.C)
        outputs = tf.keras.layers.Dense(output_dim)(x)
        outputs = tf.keras.layers.Reshape((self.S, self.S, self.B * 5 + self.C))(outputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    
    
    
    def yolo_loss(self, y_true, y_pred):
        """
                
        Args:
            y_true: Ground truth tensor
            y_pred: Prediction tensor
        """
            
        pred = tf.reshape(y_pred, [-1, self.S, self.S, self.B * 5 + self.C])
        truth = tf.reshape(y_true, [-1, self.S, self.S, self.B * 5 + self.C]) 
         # Predicted coordinates and sizes
        pred_xy = pred[:, :, :, :2*self.B]
        pred_wh = pred[:, :, :, 2*self.B:4*self.B]
        
        # Predicted confidence scores
        pred_conf = pred[:, :, :, 4*self.B:5*self.B]
        
        # Predicted class probabilities
        pred_classes = pred[:, :, :, 5*self.B:]
        
        # Ground truth coordinates and sizes
        truth_xy = truth[:, :, :, :2*self.B]
        truth_wh = truth[:, :, :, 2*self.B:4*self.B]
        
        # Ground truth confidence scores (object exists = 1, no object = 0)
        truth_conf = truth[:, :, :, 4*self.B:5*self.B]
        
        # Ground truth class probabilities
        truth_classes = truth[:, :, :, 5*self.B:]
        
        
        object_mask = tf.reduce_sum(truth_conf, axis=3, keepdims=True)
        xy_loss = self.lambda_coord * tf.reduce_sum(
            object_mask * tf.reduce_sum(tf.square(truth_xy - pred_xy), axis=3, keepdims=True)
        )
         #Width and height loss (square root of width and height)
        
        wh_loss = self.lambda_coord * tf.reduce_sum(
            object_mask * tf.reduce_sum(
                tf.square(tf.sqrt(tf.maximum(truth_wh, 1e-10)) - tf.sqrt(tf.maximum(pred_wh, 1e-10))),
                axis=3, keepdims=True
            )
        )
        # Confidence loss for cells containing objects
        conf_obj_loss = tf.reduce_sum(
            object_mask * tf.reduce_sum(tf.square(truth_conf - pred_conf), axis=3, keepdims=True)
        )
        
        # Confidence loss for cells not containing objects
        conf_noobj_loss = self.lambda_noobj * tf.reduce_sum(
            (1 - object_mask) * tf.reduce_sum(tf.square(truth_conf - pred_conf), axis=3, keepdims=True)
        )
        
        # Class prediction loss
        class_loss = tf.reduce_sum(
            object_mask * tf.reduce_sum(tf.square(truth_classes - pred_classes), axis=3, keepdims=True)
        )
        
        # Total loss
        total_loss = xy_loss + wh_loss + conf_obj_loss + conf_noobj_loss + class_loss
        
        return total_loss
      
    def compile_model(self, optimizer=None):
        """
        Compile the model with the custom YOLO loss function
        
        Args:
            optimizer: TensorFlow optimizer (default: Adam with lr=1e-4)
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            
        self.model.compile(optimizer=optimizer, loss=self.yolo_loss)
        
    def predict(self, img):
        """
        Run prediction on an image
        
        Args:
            img: Input image (numpy array)
            
        Returns:
            Processed predictions
        """
        # Preprocess image
        resized_img = cv2.resize(img, (self.img_size, self.img_size))
        normalized_img = resized_img / 255.0
        input_img = np.expand_dims(normalized_img, axis=0)
        
        # Get raw predictions
        raw_predictions = self.model.predict(input_img)
        
        # Process predictions
        return self._process_predictions(raw_predictions[0], img.shape[0], img.shape[1])
        
    def _process_predictions(self, predictions, orig_h, orig_w):
        """
        Args:
            predictions: Raw predictions from the model
            orig_h: Original image height
            orig_w: Original image width
        """
        boxes = []
        class_scores = []
        
        # Iterate through grid cells
        for i in range(self.S):
            for j in range(self.S):
                # Cell position
                cell_x = j / self.S
                cell_y = i / self.S
                
                # Class probabilities for this cell
                class_probs = predictions[i, j, self.B*5:]
                
                # Process each bounding box in the cell
                for b in range(self.B):
                    # Box confidence
                    box_confidence = predictions[i, j, 4*b + 4]
                    
                    # Class scores = box confidence * class probabilities
                    scores = box_confidence * class_probs
                    class_idx = np.argmax(scores)
                    class_score = scores[class_idx]
                    
                    # Skip if confidence is too low
                    if class_score < 0.2:
                        continue
                    
                    # Box center coordinates (relative to cell)
                    box_xy = predictions[i, j, 4*b:4*b+2]
                    
                    # Convert to absolute coordinates in the image
                    box_xy[0] = (cell_x + box_xy[0]) * orig_w
                    box_xy[1] = (cell_y + box_xy[1]) * orig_h
                    
                    # Box dimensions
                    box_wh = predictions[i, j, 4*b+2:4*b+4]
                    box_wh[0] = box_wh[0] * orig_w
                    box_wh[1] = box_wh[1] * orig_h
                    
                    # Convert to top-left and bottom-right coordinates
                    x1 = max(0, box_xy[0] - box_wh[0]/2)
                    y1 = max(0, box_xy[1] - box_wh[1]/2)
                    x2 = min(orig_w, box_xy[0] + box_wh[0]/2)
                    y2 = min(orig_h, box_xy[1] + box_wh[1]/2)
                    
                    boxes.append([x1, y1, x2, y2])
                    class_scores.append((class_idx, class_score))
        
        # Apply non-maximum suppression
        return self._non_max_suppression(boxes, class_scores)
    
    def _non_max_suppression(self, boxes, class_scores, iou_threshold=0.5):
        """
        Args:
            boxes: List of bounding boxes
            class_scores: List of (class_index, score) tuples
            iou_threshold: IoU threshold for NMS
            """
        if not boxes:
            return []
    
    # Convert to numpy array
        boxes = np.array(boxes)
    
    # Get scores only
        scores = np.array([score for _, score in class_scores])
        classes = np.array([cls_idx for cls_idx, _ in class_scores])
        
        # Sort by score
        indices = np.argsort(-scores)
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]
        
        keep_indices = []
        
        while len(boxes) > 0:
            # Keep the box with highest score
            keep_indices.append(indices[0])
            
            if len(boxes) == 1:
                break
                
            # Calculate IoU with the remaining boxes
            ious = self._calculate_iou(boxes[0], boxes[1:])
            
            # Filter out boxes with IoU > threshold
            mask = ious < iou_threshold
            boxes = boxes[1:][mask]
            scores = scores[1:][mask]
            classes = classes[1:][mask]
            indices = indices[1:][mask]
        
        # Return kept detections
        final_detections = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes[idx]
            final_detections.append({
                'box': (x1, y1, x2, y2),
                'class': classes[idx],
                'score': scores[idx]
            })
        
        return final_detections

    def _calculate_iou(self, box, boxes):
        """
        Calculate IoU between a box and an array of boxes
        
        Args:
            box: Single box [x1, y1, x2, y2]
            boxes: Array of boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Array of IoU values
        """
        # Box area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        
        # Other boxes areas
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Intersection coordinates
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        
        # Intersection area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # IoU
        return intersection / (box_area + area - intersection)
           