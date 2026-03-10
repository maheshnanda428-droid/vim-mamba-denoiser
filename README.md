# Vision Mamba Satellite Image Denoiser

This project implements a **Vision Mamba based neural network** for removing noise from satellite imagery.

## Features

* Poisson + Gaussian orbital noise simulation
* Vision Mamba architecture
* Satellite image denoising
* ONNX export for Jetson Orin Nano deployment

## Project Structure

dataset/ – dataset loading
models/ – Vision Mamba model
training/ – training pipeline
export/ – ONNX export scripts

## Installation

pip install -r requirements.txt

## Train Model

python training/train.py

## Export Model

python export/export_onnx.py
