#!/usr/bin/env python3
"""
Example Usage Scripts for ComfyUI Flame Integration
Demonstrates different ways to use the integration
"""

import sys
import asyncio
import cv2
import numpy as np
from pathlib import Path

from comfyui_client import ComfyUIClient, ComfyUIConfig


def example_1_basic_processing():
    """
    Example 1: Basic image processing with ComfyUI
    """
    print("=" * 60)
    print("Example 1: Basic Image Processing")
    print("=" * 60)

    async def process():
        # Create client
        config = ComfyUIConfig(host="127.0.0.1", port=8188)
        client = ComfyUIClient(config)

        # Connect
        print("\n1. Connecting to ComfyUI...")
        await client.connect()

        # Load workflow
        print("2. Loading workflow...")
        workflow = client.load_workflow("workflows/upscale_4k_professional.json")

        if not workflow:
            print("✗ Failed to load workflow")
            return

        # Load input image
        print("3. Loading input image...")
        image_path = "test_image.jpg"

        if not Path(image_path).exists():
            print(f"✗ Image not found: {image_path}")
            print("  Please create a test image first")
            return

        input_image = cv2.imread(image_path)

        # Process
        print("4. Processing image...")

        def progress_callback(status, current, total):
            print(f"   Progress: {status} - {current}/{total}")

        results = await client.execute_workflow(
            workflow,
            images={"input_image": input_image},
            progress_callback=progress_callback
        )

        # Save result
        if results:
            output_path = "output_upscaled_4k.jpg"
            cv2.imwrite(output_path, results[0])
            print(f"\n✓ Result saved to: {output_path}")
        else:
            print("\n✗ Processing failed")

        # Disconnect
        await client.disconnect()

    # Run
    asyncio.run(process())


def example_2_batch_processing():
    """
    Example 2: Batch process multiple images
    """
    print("=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    async def process():
        config = ComfyUIConfig(host="127.0.0.1", port=8188)
        client = ComfyUIClient(config)

        await client.connect()

        workflow = client.load_workflow("workflows/denoise_professional.json")

        if not workflow:
            print("✗ Failed to load workflow")
            return

        # List of images to process
        image_files = [
            "image1.jpg",
            "image2.jpg",
            "image3.jpg"
        ]

        # Process each image
        for i, image_file in enumerate(image_files):
            if not Path(image_file).exists():
                print(f"⚠ Skipping {image_file} (not found)")
                continue

            print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file}")

            input_image = cv2.imread(image_file)

            results = await client.execute_workflow(
                workflow,
                images={"input_image": input_image}
            )

            if results:
                output_file = f"denoised_{Path(image_file).stem}.jpg"
                cv2.imwrite(output_file, results[0])
                print(f"✓ Saved: {output_file}")

        await client.disconnect()
        print("\n✓ Batch processing complete!")

    asyncio.run(process())


def example_3_custom_parameters():
    """
    Example 3: Using custom workflow parameters
    """
    print("=" * 60)
    print("Example 3: Custom Parameters")
    print("=" * 60)

    async def process():
        config = ComfyUIConfig(host="127.0.0.1", port=8188)
        client = ComfyUIClient(config)

        await client.connect()

        workflow = client.load_workflow("workflows/color_grade_enhancement.json")

        if not workflow:
            print("✗ Failed to load workflow")
            return

        # Load input
        input_image = cv2.imread("test_image.jpg")

        if input_image is None:
            print("✗ Could not load test image")
            return

        # Define custom parameters
        custom_params = {
            "6": {  # KSampler node ID
                "denoise": 0.4,  # Higher denoise for stronger effect
                "cfg": 6.0,
                "steps": 25
            }
        }

        # Update workflow with custom parameters
        custom_workflow = client.update_workflow_params(workflow, custom_params)

        print("\nProcessing with custom parameters:")
        print(f"  Denoise: {custom_params['6']['denoise']}")
        print(f"  CFG: {custom_params['6']['cfg']}")
        print(f"  Steps: {custom_params['6']['steps']}")

        # Execute
        results = await client.execute_workflow(
            custom_workflow,
            images={"input_image": input_image}
        )

        if results:
            cv2.imwrite("output_custom_color_grade.jpg", results[0])
            print("\n✓ Result saved with custom parameters")

        await client.disconnect()

    asyncio.run(process())


def example_4_flame_pybox_simulation():
    """
    Example 4: Simulate Flame PyBox processing
    """
    print("=" * 60)
    print("Example 4: Flame PyBox Simulation")
    print("=" * 60)

    from flame_comfyui_pybox import FlameComfyUIProcessor

    # Create processor
    processor = FlameComfyUIProcessor("comfyui_config.json")

    # List available workflows
    print("\nAvailable workflows:")
    for workflow in processor.get_workflow_list():
        print(f"  - {workflow}")

    # Load test image
    test_image = cv2.imread("test_image.jpg")

    if test_image is None:
        print("\n⚠ Test image not found, skipping processing demo")
        return

    # Process single frame
    print("\nProcessing single frame...")
    workflow_name = processor.get_workflow_list()[0] if processor.get_workflow_list() else None

    if workflow_name:
        result = processor.process_single_frame(test_image, workflow_name)

        if result is not None:
            cv2.imwrite("output_pybox_simulation.jpg", result)
            print("✓ PyBox simulation complete")
        else:
            print("✗ Processing failed")


def example_5_video_frame_extraction():
    """
    Example 5: Extract and process video frames
    """
    print("=" * 60)
    print("Example 5: Video Frame Processing")
    print("=" * 60)

    async def process():
        # Open video
        video_path = "test_video.mp4"

        if not Path(video_path).exists():
            print(f"⚠ Video not found: {video_path}")
            print("  Please provide a test video")
            return

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("✗ Could not open video")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo info:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # Setup ComfyUI
        config = ComfyUIConfig(host="127.0.0.1", port=8188)
        client = ComfyUIClient(config)
        await client.connect()

        workflow = client.load_workflow("workflows/texture_detail_enhancement.json")

        if not workflow:
            print("✗ Failed to load workflow")
            return

        # Process first 10 frames as example
        print("\nProcessing first 10 frames...")
        processed_frames = []

        for i in range(min(10, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            print(f"  Frame {i+1}/10...", end=" ")

            results = await client.execute_workflow(
                workflow,
                images={"input_image": frame}
            )

            if results:
                processed_frames.append(results[0])
                print("✓")
            else:
                processed_frames.append(frame)
                print("✗")

        cap.release()

        # Save processed frames
        if processed_frames:
            print(f"\nSaving {len(processed_frames)} processed frames...")

            for i, frame in enumerate(processed_frames):
                output_path = f"frame_{i:04d}.jpg"
                cv2.imwrite(output_path, frame)

            print("✓ Frames saved")

        await client.disconnect()

    asyncio.run(process())


def example_6_workflow_comparison():
    """
    Example 6: Compare multiple workflows on same image
    """
    print("=" * 60)
    print("Example 6: Workflow Comparison")
    print("=" * 60)

    async def process():
        config = ComfyUIConfig(host="127.0.0.1", port=8188)
        client = ComfyUIClient(config)
        await client.connect()

        # Load input
        input_image = cv2.imread("test_image.jpg")

        if input_image is None:
            print("✗ Could not load test image")
            return

        # Workflows to compare
        workflows_to_test = [
            "upscale_4k_professional",
            "denoise_professional",
            "color_grade_enhancement",
            "texture_detail_enhancement"
        ]

        print(f"\nComparing {len(workflows_to_test)} workflows:\n")

        results_dict = {}

        for workflow_name in workflows_to_test:
            workflow_path = f"workflows/{workflow_name}.json"

            if not Path(workflow_path).exists():
                print(f"⚠ Skipping {workflow_name} (not found)")
                continue

            print(f"Processing with: {workflow_name}...")

            workflow = client.load_workflow(workflow_path)

            if workflow:
                results = await client.execute_workflow(
                    workflow,
                    images={"input_image": input_image}
                )

                if results:
                    output_path = f"comparison_{workflow_name}.jpg"
                    cv2.imwrite(output_path, results[0])
                    results_dict[workflow_name] = output_path
                    print(f"  ✓ Saved: {output_path}")

        print(f"\n✓ Comparison complete! Generated {len(results_dict)} outputs")
        print("\nResults:")
        for name, path in results_dict.items():
            print(f"  {name}: {path}")

        await client.disconnect()

    asyncio.run(process())


def main():
    """Main function - run examples"""
    print("\n" + "=" * 60)
    print("ComfyUI Flame Integration - Example Usage")
    print("=" * 60)

    examples = {
        "1": ("Basic Image Processing", example_1_basic_processing),
        "2": ("Batch Processing", example_2_batch_processing),
        "3": ("Custom Parameters", example_3_custom_parameters),
        "4": ("Flame PyBox Simulation", example_4_flame_pybox_simulation),
        "5": ("Video Frame Processing", example_5_video_frame_extraction),
        "6": ("Workflow Comparison", example_6_workflow_comparison),
    }

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\n  0. Exit")

    while True:
        choice = input("\nSelect example (0-6): ").strip()

        if choice == "0":
            print("Goodbye!")
            break

        if choice in examples:
            name, func = examples[choice]
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}\n")

            try:
                func()
            except Exception as e:
                print(f"\n✗ Error running example: {e}")
                import traceback
                traceback.print_exc()

            print("\n" + "=" * 60)
        else:
            print("Invalid choice. Please select 0-6.")


if __name__ == "__main__":
    main()
