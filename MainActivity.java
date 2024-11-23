package com.example.sketchtoimage;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sketchtoimage.ml.SketchToImageModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final String TAG = "SketchToImage";

    ImageView inputImageView, outputImageView;
    Button processButton, chooseImageButton;
    ExecutorService executorService;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inputImageView = findViewById(R.id.inputImageView);
        outputImageView = findViewById(R.id.outputImageView);
        processButton = findViewById(R.id.processButton);
        chooseImageButton = findViewById(R.id.chooseImageButton);

        // Initialize ExecutorService for background tasks
        executorService = Executors.newSingleThreadExecutor();

        // Button to choose image from gallery
        chooseImageButton.setOnClickListener(view -> openImageChooser());

        // Process the selected image in the background
        processButton.setOnClickListener(view -> {
            Bitmap inputBitmap = ((BitmapDrawable) inputImageView.getDrawable()).getBitmap();
            if (inputBitmap != null) {
                executorService.submit(() -> processImage(inputBitmap));
            } else {
                Toast.makeText(MainActivity.this, "Please select an image first", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void openImageChooser() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                inputImageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    private void processImage(Bitmap inputBitmap) {
        try {
            // Initialize the TFLite model
            SketchToImageModel model = SketchToImageModel.newInstance(getApplicationContext());

            // Preprocess the input bitmap for the model
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, 178, 218, true);
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 218, 178, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedBitmap);
            inputFeature0.loadBuffer(byteBuffer);

            // Log normalized input data
            logByteBuffer(byteBuffer);

            // Process the image with the TFLite model
            SketchToImageModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Log output tensor data
            logTensorBuffer(outputFeature0);

            // Convert the model output to a bitmap
            Bitmap outputBitmap = convertTensorBufferToBitmap(outputFeature0, 178, 218);

            // Update UI with the output image
            runOnUiThread(() -> {
                outputImageView.setImageBitmap(outputBitmap);
                Toast.makeText(MainActivity.this, "Processing Complete", Toast.LENGTH_SHORT).show();
            });

            // Release the model resources
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
            runOnUiThread(() -> Toast.makeText(MainActivity.this, "Error processing image", Toast.LENGTH_SHORT).show());
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 218 * 178 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[218 * 178];
        bitmap.getPixels(intValues, 0, 178, 0, 0, 178, 218);

        for (int pixelValue : intValues) {
            byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f); // Normalize Red
            byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);  // Normalize Green
            byteBuffer.putFloat((pixelValue & 0xFF) / 255.0f);         // Normalize Blue
        }

        return byteBuffer;
    }

    private Bitmap convertTensorBufferToBitmap(TensorBuffer tensorBuffer, int width, int height) {
        float[] floatArray = tensorBuffer.getFloatArray();
        int[] pixels = new int[width * height];

        for (int i = 0; i < pixels.length; i++) {
            int r = (int) (Math.min(Math.max(floatArray[i * 3], 0.0f), 1.0f) * 255);
            int g = (int) (Math.min(Math.max(floatArray[i * 3 + 1], 0.0f), 1.0f) * 255);
            int b = (int) (Math.min(Math.max(floatArray[i * 3 + 2], 0.0f), 1.0f) * 255);

            pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
    }

    private void logByteBuffer(ByteBuffer byteBuffer) {
        byteBuffer.rewind();
        for (int i = 0; i < byteBuffer.capacity(); i += 4) {
            Log.d(TAG, "Input ByteBuffer Value: " + byteBuffer.getFloat(i));
        }
    }

    private void logTensorBuffer(TensorBuffer tensorBuffer) {
        float[] outputValues = tensorBuffer.getFloatArray();
        for (int i = 0; i < outputValues.length; i++) {
            Log.d(TAG, "Output Tensor Value: " + outputValues[i]);
        }
    }
}
