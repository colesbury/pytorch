#pragma once

#include <THPP/Tensor.hpp>

namespace torch { namespace nn {

void Abs_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void Abs_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput);


void AbsCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage);


void AbsCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage);


void BCECriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage, thpp::Tensor* weights);


void BCECriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage, thpp::Tensor* weights);


void ClassNLLCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage, thpp::Tensor* weights, thpp::Tensor* total_weight);


void ClassNLLCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage, thpp::Tensor* weights, thpp::Tensor* total_weight);


void SpatialClassNLLCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage, thpp::Tensor* weights, thpp::Tensor* total_weight);


void SpatialClassNLLCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage, thpp::Tensor* weights, thpp::Tensor* total_weight);


void ELU_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double alpha, bool inplace);


void ELU_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output, double alpha, bool inplace);


void DistKLDivCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage);


void DistKLDivCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage);


void HardShrink_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double lambda);


void HardShrink_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, double lambda);


void HardTanh_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double min_val, double max_val, bool inplace);


void HardTanh_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, double min_val, double max_val, bool inplace);


void L1Cost_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void L1Cost_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput);


void LeakyReLU_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double negval, bool inplace);


void LeakyReLU_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, double negval, bool inplace);


void LogSigmoid_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* buffer);


void LogSigmoid_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* buffer);


void LogSoftMax_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void LogSoftMax_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output);


void MarginCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage, double margin);


void MarginCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage, double margin);


void SoftMarginCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage);


void SoftMarginCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage);


void MSECriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage);


void MSECriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage);


void MultiLabelMarginCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, thpp::Tensor* isTarget, bool sizeAverage);


void MultiLabelMarginCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, thpp::Tensor* isTarget, bool sizeAverage);


void MultiMarginCriterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage, int p, thpp::Tensor* weights, double margin);


void MultiMarginCriterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage, int p, thpp::Tensor* weights, double margin);


void PReLU_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, long nOutputPlane);


void PReLU_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, long nOutputPlane);


void PReLU_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* gradWeight, thpp::Tensor* gradWeightBuf, thpp::Tensor* gradWeightBuf2, long nOutputPlane, double scale);


void Linear_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* addBuffer);


void Linear_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight);


void Linear_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* addBuffer, double scale);


void RReLU_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* noise, double lower, double upper, bool train, bool inplace, THGenerator* generator);


void RReLU_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* noise, double lower, double upper, bool train, bool inplace);


void Sigmoid_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void Sigmoid_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output);


void SmoothL1Criterion_updateOutput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* output, bool sizeAverage);


void SmoothL1Criterion_updateGradInput(thpp::Tensor* input, thpp::Tensor* target, thpp::Tensor* gradInput, bool sizeAverage);


void SoftMax_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void SoftMax_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output);


void SoftPlus_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double beta, double threshold);


void SoftPlus_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output, double beta, double threshold);


void SoftShrink_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double lambda);


void SoftShrink_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, double lambda);


void SparseLinear_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias);


void SparseLinear_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* weight, thpp::Tensor* bias, double weightDecay, double scale);


void Sqrt_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double eps);


void Sqrt_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output);


void Square_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void Square_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput);


void Tanh_updateOutput(thpp::Tensor* input, thpp::Tensor* output);


void Tanh_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* output);


void Threshold_updateOutput(thpp::Tensor* input, thpp::Tensor* output, double threshold, double val, bool inplace);


void Threshold_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, double threshold, double val, bool inplace);


void TemporalConvolution_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, int kW, int dW, int inputFrameSize, int outputFrameSize);


void TemporalConvolution_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, int kW, int dW);


void TemporalConvolution_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, int kW, int dW, double scale);


void TemporalMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int kW, int dW);


void TemporalMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int kW, int dW);


void TemporalSubSampling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, int kW, int dW, int inputFrameSize);


void TemporalSubSampling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, int kW, int dW);


void TemporalSubSampling_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, int kW, int dW, double scale);


void BatchNormalization_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* running_mean, thpp::Tensor* running_var, thpp::Tensor* save_mean, thpp::Tensor* save_std, bool train, double momentum, double eps);


void BatchNormalization_backward(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* weight, thpp::Tensor* running_mean, thpp::Tensor* running_var, thpp::Tensor* save_mean, thpp::Tensor* save_std, bool train, double scale, double eps);


void SpatialConvolutionMap_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* connTable, int nInputPlane, int nOutputPlane, int dW, int dH);


void SpatialConvolutionMap_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* connTable, int nInputPlane, int nOutputPlane, int dW, int dH);


void SpatialConvolutionMap_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* connTable, int nInputPlane, int nOutputPlane, int dW, int dH, double scale);


void SpatialConvolutionMM_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);


void SpatialConvolutionMM_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);


void SpatialConvolutionMM_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, double scale);


void SpatialConvolutionLocal_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, long inputWidth, long inputHeight, long outputWidth, long outputHeight);


void SpatialConvolutionLocal_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, long inputWidth, long inputHeight, long outputWidth, long outputHeight);


void SpatialConvolutionLocal_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, long inputWidth, long inputHeight, long outputWidth, long outputHeight, double scale);


void SpatialAdaptiveMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int owidth, int oheight);


void SpatialAdaptiveMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices);


void SpatialAveragePooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);


void SpatialAveragePooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);


void SpatialFractionalMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int outputW, int outputH, int poolSizeW, int poolSizeH, thpp::Tensor* indices, thpp::Tensor* randomSamples);


void SpatialFractionalMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int outputW, int outputH, int poolSizeW, int poolSizeH, thpp::Tensor* indices);


void SpatialFullConvolution_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* columns, thpp::Tensor* ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH);


void SpatialFullConvolution_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH);


void SpatialFullConvolution_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* columns, thpp::Tensor* ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, double scale);


void SpatialFullConvolutionMap_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* connTable, int nInputPlane, int nOutputPlane, int dW, int dH);


void SpatialFullConvolutionMap_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* connTable, int nInputPlane, int nOutputPlane, int dW, int dH);


void SpatialFullConvolutionMap_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* connTable, int nInputPlane, int nOutputPlane, int dW, int dH, double scale);


void SpatialDilatedConvolution_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* columns, thpp::Tensor* ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);


void SpatialDilatedConvolution_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);


void SpatialDilatedConvolution_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* columns, thpp::Tensor* ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, double scale);


void SpatialMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);


void SpatialMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);


void SpatialDilatedMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode);


void SpatialDilatedMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode);


void SpatialMaxUnpooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int owidth, int oheight);


void SpatialMaxUnpooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int owidth, int oheight);


void SpatialSubSampling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, int kW, int kH, int dW, int dH);


void SpatialSubSampling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, int kW, int kH, int dW, int dH);


void SpatialSubSampling_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, int kW, int kH, int dW, int dH, double scale);


void SpatialUpSamplingNearest_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int scale_factor);


void SpatialUpSamplingNearest_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int scale_factor);


void SpatialUpSamplingBilinear_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int outputHeight, int outputWidth);


void SpatialUpSamplingBilinear_updateGradInput(thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int nbatch, int nchannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth);


void VolumetricAveragePooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int kT, int kW, int kH, int dT, int dW, int dH);


void VolumetricAveragePooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int kT, int kW, int kH, int dT, int dW, int dH);


void VolumetricConvolution_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int dT, int dW, int dH, int pT, int pW, int pH);


void VolumetricConvolution_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* finput, int dT, int dW, int dH, int pT, int pW, int pH);


void VolumetricConvolution_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, double scale);


void VolumetricConvolutionMM_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);


void VolumetricConvolutionMM_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* finput, thpp::Tensor* fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);


void VolumetricConvolutionMM_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, double scale);


void VolumetricFullConvolution_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH);


void VolumetricFullConvolution_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* finput, thpp::Tensor* fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH);


void VolumetricFullConvolution_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* finput, thpp::Tensor* fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, double scale);


void VolumetricDilatedConvolution_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* weight, thpp::Tensor* bias, thpp::Tensor* columns, thpp::Tensor* ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);


void VolumetricDilatedConvolution_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* weight, thpp::Tensor* gradColumns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);


void VolumetricDilatedConvolution_accGradParameters(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradWeight, thpp::Tensor* gradBias, thpp::Tensor* columns, thpp::Tensor* ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, double scale);


void VolumetricMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode);


void VolumetricMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode);


void VolumetricDilatedMaxPooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode);


void VolumetricDilatedMaxPooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode);


void VolumetricMaxUnpooling_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);


void VolumetricMaxUnpooling_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);


void SpatialReflectionPadding_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int pad_l, int pad_r, int pad_t, int pad_b);


void SpatialReflectionPadding_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int pad_l, int pad_r, int pad_t, int pad_b);


void SpatialReplicationPadding_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int pad_l, int pad_r, int pad_t, int pad_b);


void SpatialReplicationPadding_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int pad_l, int pad_r, int pad_t, int pad_b);


void VolumetricReplicationPadding_updateOutput(thpp::Tensor* input, thpp::Tensor* output, int pleft, int pright, int ptop, int pbottom, int pfront, int pback);


void VolumetricReplicationPadding_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, int pleft, int pright, int ptop, int pbottom, int pfront, int pback);


void SpatialCrossMapLRN_updateOutput(thpp::Tensor* input, thpp::Tensor* output, thpp::Tensor* scale, int size, double alpha, double beta, double k);


void SpatialCrossMapLRN_updateGradInput(thpp::Tensor* input, thpp::Tensor* gradOutput, thpp::Tensor* gradInput, thpp::Tensor* scale, thpp::Tensor* output, int size, double alpha, double beta, double k);


}} // namespace torch::nn
