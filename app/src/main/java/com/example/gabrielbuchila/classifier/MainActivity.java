package com.example.gabrielbuchila.classifier;


import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.camerakit.CameraKitView;


import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    private static final int DstWidth = 224;
    private static final int DstHeight = DstWidth;

    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private ImageClassifier classifier;
    private object_classified results;
    private Executor executor = Executors.newSingleThreadExecutor();

    private TextView textViewResult;
    private Button btnDetectObject, btnToggleCamera;
    private ImageView imageViewResult;
    private CameraKitView cameraView;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraView = findViewById(R.id.cameraView);


        imageViewResult = (ImageView) findViewById(R.id.imageViewResult);
        textViewResult = (TextView) findViewById(R.id.textViewResult);

        textViewResult.setMovementMethod(new ScrollingMovementMethod());


        btnToggleCamera = (Button) findViewById(R.id.btnToggleCamera);
        btnDetectObject = (Button) findViewById(R.id.btnDetectObject);


        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.toggleFacing();
            }
        });


        btnDetectObject.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage(new CameraKitView.ImageCallback() {
                    @Override
                    public void onImage(CameraKitView cameraKitView, final byte[] capturedImage) {
                        try {
                            /*****************************
                             *(BitmapFactory.decodeByteArray)
                             *******************************
                             * Parameters:
                             * - capturedImage : byte array of compressed image data
                             * - offset : offset into imageData for where the decoder should begin parsing
                             * - capturedImage.length : the number of bytes, beginning at offset, to parse
                             * Returns:
                             * - The decoded bitmap, or null if the image could not be decoded.
                             ***************************************************************/
                            Bitmap bitmap = BitmapFactory.decodeByteArray(capturedImage, 0, capturedImage.length);

                            /********************************
                             * (Bitmap.createScaledBitmap)
                             ********************************
                             * Parameters:
                             * - bitmap : The source bitmap
                             * - DstWidth : The new bitmap's desired width.
                             * - filter : true if the source should be filtered.
                             * Returns:
                             * - The new scaled bitmap or the source bitmap if no scaling is required..
                             */
                            bitmap = Bitmap.createScaledBitmap(bitmap, DstWidth, DstWidth, false);

                            /********************************
                             * (imageViewResult.setImageBitmap)
                             ********************************
                             * Parameters:
                             * - bitmap : The bitmap to set
                             */

                            imageViewResult.setImageBitmap(bitmap);

                            results = classifier.recognizeImage(bitmap);

                            textViewResult.setText("The object is :" + results.getTitle() + "\n" + "The probability is :  " + results.getConfidence() + " %");

                        }catch (final Exception e) {
                            e.printStackTrace();
                        }
                    }
                });

            }
        });

        textViewResult.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // ATTENTION: This was auto-generated to handle app links.
                String url = "https://en.wikipedia.org/wiki/"+results.getTitle();

                Intent i = new Intent(Intent.ACTION_VIEW);
                i.setData(Uri.parse(url));
                startActivity(i);
            }
        });


        initTensorFlowAndLoadModel();
    }


    @Override
    protected void onStart() {
        super.onStart();
        cameraView.onStart();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.onResume();
    }

    @Override
    protected void onPause() {
        cameraView.onPause();
        super.onPause();
    }

    @Override
    protected void onStop() {
        cameraView.onStop();
        super.onStop();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        cameraView.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });


     }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier=new ImageClassifier(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            DstWidth,
                            INPUT_NAME,
                            OUTPUT_NAME
                            );
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDetectObject.setVisibility(View.VISIBLE);
            }
        });
    }

}

