# Verifying GPU Availability on macOS

Command-line instructions to check for and verify the availability of a Graphics Processing Unit (GPU) on a Mac.

## Using System Commands

You can quickly check your system's hardware profile to see if a GPU is installed. The primary command for this is `system_profiler`.

### Apple Silicon (M1/M2/M3)

For Macs with Apple Silicon, use either of the following commands in your terminal:

*   To get a general display and graphics report:
    ```sh
    system_profiler SPDisplaysDataType
    ```
*   For a more GPU-specific report (this uses a private framework and may change in future macOS versions):
    ```sh
    /System/Library/PrivateFrameworks/AppleGPUWrangler.framework/Versions/A/Resources/gpu-info
    ```

### Intel-based Macs

For older Intel-based Macs, the `system_profiler` command will list any discrete or integrated GPUs.

```sh
system_profiler SPDisplaysDataType
```

## Using Python Deep Learning Frameworks

If you are a developer, you can also verify that a specific deep learning framework can detect and use the GPU. The following snippets will list all computational devices available to the framework, including GPUs.

### TensorFlow

```python
import tensorflow as tf

# This will return a list of physical GPU devices visible to TensorFlow
print(tf.config.list_physical_devices('GPU'))
```

### JAX

```python
import jax

# This will return a list of all available JAX devices (CPU, GPU, etc.)
print(jax.devices())
```

# Additional Resources

*   [Jupyter notebooks for the code samples](https://github.com/fchollet/deep-learning-with-python-notebooks/tree/third-edition)