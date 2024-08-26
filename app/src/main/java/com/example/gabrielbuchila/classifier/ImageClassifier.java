
package com.example.gabrielbuchila.classifier;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Vector;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** A classifier specialized to label images using TensorFlow. */
public class ImageClassifier {
    /*************
    LOCAL VARIABLE
     **************/
    // MACRO
    private static final float THRESHOLD = 0.1f;
    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private ImageClassifier() {}

    /*
     * Initializes a native TensorFlow session for classifying images.
     */


    /*****************************
     *(ImageClassifier)
     *
     * Parameters:
     * - new_assetManager : The asset manager to be used to load assets
     * - new_modelFilename : The filepath of the model GraphDef protocol buffer
     * - new_labelFilename : The filepath of label file for classes.
     * - new_inputSize : The input size. A square image of inputSize x inputSize is assumed.
     * - new_inputName : The label of the image input node.
     * - new_outputName : The label of the output node.
     * Returns:
     * - no returns
     ***************************************************************/

    ImageClassifier(AssetManager new_assetManager, String new_modelFilename,
                    String new_labelFilename, int new_inputSize,
                    String new_inputName, String new_outputName)
    {

        this.inputName = new_inputName;
        this.outputName = new_outputName;
        this.inputSize = new_inputSize;

        this.inferenceInterface = new TensorFlowInferenceInterface(new_assetManager, new_modelFilename);
        final Operation operation = this.inferenceInterface.graphOperation(outputName);  // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final int numClasses = (int) operation.output(0).shape().size(1);

        // Pre-allocate buffers.
        this.outputNames = new String[] {outputName};
        this.intValues = new int[inputSize * inputSize];
        this.floatValues = new float[inputSize * inputSize * 3];
        this.outputs = new float[numClasses];
        String actualFilename = new_labelFilename.split("file:///android_asset/")[1];  //reading labels

        //read all labels from file and added to labels local buffer
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new_assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                this.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("have a problem to read a labels !" , e);
        }

    }

    public  object_classified recognizeImage(final Bitmap bitmap) {

        /*
         * Here starting the bitmap processing
         * Preprocess the image data from 0-255 int to normalized float based
         */

        /********************************
         * (bitmap.getPixels)
         ********************************
         * Parameters:
         * - intValues : The array to receive the bitmap's colors
         * - offset : The first index to write into pixels[]
         * - bitmap.getWidth() : The number of entries in pixels[] to skip between rows (must be >= bitmap's width). Can be negative.
         * - x : The x coordinate of the first pixel to read from the bitmap.
         * - y : The y coordinate of the first pixel to read from the bitmap.
         * - bitmap.getWidth() : The number of pixels to read from each row.
         * - bitmap.getHeight(): The number of rows to read.
         * Returns:
         * - The new scaled bitmap or the source bitmap if no scaling is required..
         */

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());


        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = Color.red(val);
            floatValues[i * 3 + 1] = Color.green(val);
            floatValues[i * 3 + 2] = Color.blue(val);
            }



        // Copy the input data into TensorFlow.
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);

        // Run the inference call.
        inferenceInterface.run(outputNames, logStats);

        // Copy the output Tensor back into the output array.
        inferenceInterface.fetch(outputName, outputs);


        // Use prority queue for find the best result
        PriorityQueue<object_classified> pq =  new PriorityQueue<object_classified>(3, new Comparator<object_classified>() {
                            @Override
                            public int compare(object_classified o1, object_classified o2) {
                                // put the high confidence at the head of the queue this is the motive because put invers .
                                return Float.compare(o2.getConfidence(), o1.getConfidence());
                            }
                        });

        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                object_classified aux=new object_classified();
                aux.set_id("" + i);
                aux.set_title(labels.size() > i ? labels.get(i) : "unknown");
                aux.set_confidence(outputs[i]);
                pq.add(aux);
            }
        }

        object_classified object =pq.poll();


        return object;
    }


    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }


    public void close() {
        inferenceInterface.close();
    }

}