#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorMath.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

void THDTensor_(fill)(THDTensor *tensor, real value) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorFill,
      tensor,
      value
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(zero)(THDTensor *r) {
  THDTensor_(fill)(r, 0);
}

void THDTensor_(zeros)(THDTensor *tensor, THLongStorage *size) {
  THDTensor_(resize)(tensor, size, nullptr);
  THDTensor_(zero)(tensor);
}

void THDTensor_(ones)(THDTensor *tensor, THLongStorage *size) {
  THDTensor_(resize)(tensor, size, nullptr);
  THDTensor_(fill)(tensor, 1);
}

ptrdiff_t THDTensor_(numel)(THDTensor *t) {
  return THDTensor_(nElement)(t);
}

void THDTensor_(diag)(THDTensor *r_, THDTensor *t, int k) {
  THArgCheck(THDTensor_(nDimension)(t) == 1 || THDTensor_(nDimension)(t) == 2,
      1, "matrix or a vector expected");

  if (THDTensor_(nDimension)(t) == 1) {
    long t_size = THDTensor_(size)(t, 0);
    long sz = t_size + (k >= 0 ? k : -k);

    THDTensor_(resize2d)(r_, sz, sz);
    THDTensor_(zero)(r_);
  } else {
    long sz;
    if (k >= 0)
      sz = std::min(THDTensor_(size)(t, 0), THDTensor_(size)(t, 1)-k);
    else
      sz = std::min(THDTensor_(size)(t, 0)+k, THDTensor_(size)(t, 1));
    THDTensor_(resize1d)(r_, sz);
  }
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDiag, r_, t, k),
    THDState::s_current_worker
  );
}

void THDTensor_(eye)(THDTensor *r, long n, long m) {
  THArgCheck(n > 0, 1, "invalid argument");

  if (m <= 0)
    m = n;

  THDTensor_(resize2d)(r, n, m);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorEye, r, n, m),
    THDState::s_current_worker
  );
}


void THDTensor_(range)(THDTensor *r_, accreal xmin,
                              accreal xmax, accreal step) {
  THArgCheck(step > 0 || step < 0, 3, "step must be a non-null number");
  THArgCheck(((step > 0) && (xmax >= xmin)) || ((step < 0) && (xmax <= xmin)),
              2, "upper bound and larger bound incoherent with step sign");

  ptrdiff_t size = static_cast<ptrdiff_t>((((xmax - xmin) / step) + 1));

  if (THDTensor_(nElement)(r_) != size)
    THDTensor_(resize1d)(r_, size);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRange, r_, xmin, xmax, step),
    THDState::s_current_worker
  );
}

void THDTensor_(randperm)(THDTensor *r_, THDGenerator *_generator, long n) {
  THArgCheck(n > 0, 1, "must be strictly positive");
  THDTensor_(resize1d)(r_, n);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRange, r_, _generator, n),
    THDState::s_current_worker
  );
}

void THDTensor_(reshape)(THDTensor *r_, THDTensor *t, THLongStorage *size) {
  THDTensor_(resize)(r_, size, NULL);
  THDTensor_(copy)(r_, t);
}

void THDTensor_(sort)(THDTensor *rt_, THDLongTensor *ri_,
                             THDTensor *t, int dimension,
                             int descendingOrder) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t),
      2, "invalid dimension %d", dimension + TH_INDEX_BASE);

  THDTensor_(resizeAs)(rt_, t);
  THDTensor_(copy)(rt_, t);

  {
    THLongStorage *size = THDTensor_(newSizeOf)(t);
    THDLongTensor_resize(ri_, size, NULL);
    THLongStorage_free(size);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSort, rt_, ri_, t, dimension, descendingOrder),
    THDState::s_current_worker
  );
}

void THDTensor_(topk)(THDTensor *rt_, THDLongTensor *ri_,
                      THDTensor *t, long k, int dim,
                      int dir, int sorted) {
  int numDims = THDTensor_(nDimension)(t);
  THArgCheck(dim >= 0 && dim < numDims, 3, "dim not in range");

  long sliceSize = THDTensor_(size)(t, dim);
  THArgCheck(k > 0 && k <= sliceSize, 2, "k not in range for dimension");

  THLongStorage *topKSize = THDTensor_(newSizeOf)(t);
  THLongStorage_set(topKSize, dim, k);
  THDTensor_(resize)(rt_, topKSize, NULL);
  THDLongTensor_resize(ri_, topKSize, NULL);
  THLongStorage_free(topKSize);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTopk, rt_, ri_, t, k, dim, dir, sorted),
    THDState::s_current_worker
  );
}

void THDTensor_(tril)(THDTensor *r_, THDTensor *t, long k) {
  THArgCheck(THDTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THDTensor_(resizeAs)(r_, t);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTril, r_, t, k),
    THDState::s_current_worker
  );
}

void THDTensor_(triu)(THDTensor *r_, THDTensor *t, long k) {
  THArgCheck(THDTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THDTensor_(resizeAs)(r_, t);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTriu, r_, t, k),
    THDState::s_current_worker
  );
}
void THDTensor_(cat)(THDTensor *r_, THDTensor *ta, THDTensor *tb, int dimension) {
  THDTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THDTensor_(catArray)(r_, inputs, 2, dimension);
}

void THDTensor_(catArray)(THDTensor *result, THDTensor **inputs,
                          int numInputs, int dimension) {
  THLongStorage *size;
  long offset;
  int ndim = dimension + 1;
  int ldimension = dimension;
  bool allEmpty = true;
  for (int i = 0; i < numInputs; i++)
    ndim = std::max(ndim, inputs[i]->nDimension);

  if (dimension == -2)
    ldimension = ndim ? (ndim - 1) : 0;

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(ldimension >= 0, 4, "invalid dimension %d", dimension + TH_INDEX_BASE);

  size = THLongStorage_newWithSize(ndim);

  for (int i = 0; i < ndim; i++) {
    long dimSize = i < inputs[0]->nDimension ?
                   inputs[0]->size[i] :
                   std::min(inputs[0]->nDimension, 1);
    if (i == ldimension) {
      for (int j = 1; j < numInputs; j++) {
        dimSize += i < inputs[j]->nDimension ?
                   inputs[j]->size[i] :
                   std::min(inputs[j]->nDimension, 1);
      }
    } else {
      for (int j = 1; j < numInputs; j++) {
        long sz = i < inputs[j]->nDimension ?
                  inputs[j]->size[i] :
                  std::min(inputs[j]->nDimension, 1);
        if (dimSize != sz && dimSize && sz) {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        } else if (!dimSize) {
          dimSize = sz;
        }
      }
    }
    allEmpty = allEmpty && !dimSize;
    size->data[i] = dimSize;
  }

  if (!allEmpty) {
    THDTensor_(resize)(result, size, NULL);
    std::vector<THDTensor*> inputs_vec(inputs, inputs + numInputs);

    // There's no need to send numInputs,
    // since sending inputs_vec does this implicitly
    masterCommandChannel->sendMessage(
      packMessage(Functions::tensorCatArray, result, inputs_vec, dimension),
      THDState::s_current_worker
    );
  }

  THLongStorage_free(size);
}

int THDTensor_(equal)(THDTensor *ta, THDTensor *tb) {
  if (!THDTensor_(isSameSizeAs)(ta, tb))
    return 0;
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorEqual, ta, tb),
    THDState::s_current_worker
  );
  return receiveValueFromWorker<int>(ta->storage->node_id);
}

void THDTensor_(tpow)(THDTensor *r_, real value, THDTensor *t) {
  THDTensor_(resizeAs)(r_, t);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorEqual, r_, t, value),
    THDState::s_current_worker
  );
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME,UPPNAME)                                \
  void THDTensor_(NAME##Value)(THDByteTensor *r_, THDTensor* t, real value) { \
    THDByteTensor__resize(r_, t->nDimension, t->size, NULL);                  \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##Value, r_, t, value),           \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \
  void THDTensor_(NAME##ValueT)(THDTensor* r_, THDTensor* t, real value)  {   \
    THDTensor_(_resize)(r_, t->nDimension, t->size, NULL);                    \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##ValueT, r_, t, value),          \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \
  void THDTensor_(NAME##Tensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb) { \
    THDByteTensor__resize(r_, ta->nDimension, ta->size, NULL);                \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##Tensor, r_, ta, tb),            \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \
  void THDTensor_(NAME##TensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb) { \
    THDTensor_(_resize)(r_, ta->nDimension, ta->size, NULL);                  \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##TensorT, r_, ta, tb),           \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \


TENSOR_IMPLEMENT_LOGICAL(lt,Lt)
TENSOR_IMPLEMENT_LOGICAL(gt,Lt)
TENSOR_IMPLEMENT_LOGICAL(le,Le)
TENSOR_IMPLEMENT_LOGICAL(ge,Ge)
TENSOR_IMPLEMENT_LOGICAL(eq,Eq)
TENSOR_IMPLEMENT_LOGICAL(ne,Ne)

#undef TENSOR_IMPLEMENT_LOGICAL

#define TENSOR_IMPLEMENT_POINTWISE_FUNCTION(NAME, UPPNAME)    \
  void THDTensor_(NAME)(THDTensor *r_, THDTensor *t) {        \
    THDTensor_(resizeAs)(r_, t);                              \
    masterCommandChannel->sendMessage(                        \
      packMessage(Functions::tensor##UPPNAME, r_, t),         \
      THDState::s_current_worker                              \
    );                                                        \
  }                                                           \

#define TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(NAME, UPPNAME)              \
  void THDTensor_(NAME)(THDTensor *r_, THDTensor *t, real value) {            \
    THDTensor_(resizeAs)(r_, t);                                              \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME, r_, t, value),                  \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \


#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) ||\
    defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(abs,Abs)
#endif

#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sigmoid,Sigmoid)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(log,Log)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(log1p,Log1p)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(exp,Exp)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(cos,Cos)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(acos,Acos)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(cosh,Cosh)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sin,Sin)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(asin,Asin)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sinh,Sinh)

TENSOR_IMPLEMENT_POINTWISE_FUNCTION(tan,Tan)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(atan,Atan)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(tanh,Tanh)
TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(pow,Pow)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sqrt,Sqrt)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(rsqrt,Rsqrt)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(ceil,Ceil)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(floor,Floor)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(round,Round)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(trunc,Trunc)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(frac,Frac)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(neg,Neg)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(cinv,Cinv)

#undef TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION
#undef TENSOR_IMPLEMENT_POINTWISE_FUNCTION

void THDTensor_(atan2)(THDTensor *r_, THDTensor *tx, THDTensor *ty) {
  THDTensor_(resizeAs)(r_, tx);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAtan2, r_, tx, ty),
    THDState::s_current_worker
  );
}

void THDTensor_(lerp)(THDTensor *r_, THDTensor *a, THDTensor *b, real weight) {
  THArgCheck(THDTensor_(nElement)(a) == THDTensor_(nElement)(b), 2,
             "sizes do not match");
  THDTensor_(resizeAs)(r_, a);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLerp, r_, a, b, weight),
    THDState::s_current_worker
  );
}

void THDTensor_(mean)(THDTensor *r_, THDTensor *t, int dimension) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 2,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMean, r_, t, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(std)(THDTensor *r_, THDTensor *t, int dimension, int flag) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorStd, r_, t, dimension, flag),
    THDState::s_current_worker
  );
}

void THDTensor_(var)(THDTensor *r_, THDTensor *t, int dimension, int flag) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorVar, r_, t, dimension, flag),
    THDState::s_current_worker
  );
}

void THDTensor_(norm)(THDTensor *r_, THDTensor *t, real value, int dimension) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorNorm, r_, t, dimension, value),
    THDState::s_current_worker
  );
}

accreal THDTensor_(normall)(THDTensor *tensor, real value) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorNormall, tensor, value),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

void THDTensor_(renorm)(THDTensor *res, THDTensor *src, real value,
                        int dimension, real maxnorm) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(src), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THDTensor_(nDimension)(src) > 1, 1,
             "need at least 2 dimensions, got %d dimensions",
             THDTensor_(nDimension)(src));

  THDTensor_(resizeAs)(res, src);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRenorm, res, src, dimension, value, maxnorm),
    THDState::s_current_worker
  );
}

accreal THDTensor_(dist)(THDTensor *tensor, THDTensor *src, real value) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDist, tensor, src, value),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(meanall)(THDTensor *tensor) {
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMeanall, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(varall)(THDTensor *tensor) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorVarall, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(stdall)(THDTensor *tensor) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorStdall, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

void THDTensor_(linspace)(THDTensor *r_, real a, real b, long n) {
  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THDTensor_(nElement)(r_) != n) {
    THDTensor_(resize1d)(r_, n);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLinspace, r_, n, a, b),
    THDState::s_current_worker
  );
}

void THDTensor_(logspace)(THDTensor *r_, real a, real b, long n) {
  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THDTensor_(nElement)(r_) != n) {
    THDTensor_(resize1d)(r_, n);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogspace, r_, n, a, b),
    THDState::s_current_worker
  );
}

void THDTensor_(rand)(THDTensor *r_, THDGenerator *_generator,
                      THLongStorage *size) {
  THDTensor_(resize)(r_, size, NULL);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRand, r_, _generator, size),
    THDState::s_current_worker
  );
}

void THDTensor_(randn)(THDTensor *r_, THDGenerator *_generator,
                       THLongStorage *size) {
  THDTensor_(resize)(r_, size, NULL);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRandn, r_, _generator, size),
    THDState::s_current_worker
  );
}

void THDTensor_(histc)(THDTensor *hist, THDTensor *tensor, long nbins,
                       real minvalue, real maxvalue) {
  THDTensor_(resize1d)(hist, nbins);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorHistc, hist, tensor, nbins, minvalue, maxvalue),
    THDState::s_current_worker
  );
}

void THDTensor_(bhistc)(THDTensor *hist, THDTensor *tensor, long nbins,
                        real minvalue, real maxvalue) {
  THArgCheck(THDTensor_(nDimension)(tensor) < 3, 2,
             "invalid dimension %d, the input must be a 2d tensor",
             THDTensor_(nDimension)(tensor));

  int dimension = 1;
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(tensor), 2,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THDTensor_(resize2d)(hist, tensor->size[0], nbins);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorBhistc, hist, tensor, nbins, minvalue, maxvalue),
    THDState::s_current_worker
  );
}

#endif // defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)

#if defined(TH_REAL_IS_BYTE)

int THDTensor_(logicalall)(THDTensor *tensor) {
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogicalall, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<int>(THDState::s_current_worker);
}

int THDTensor_(logicalany)(THDTensor *tensor) {
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogicalany, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<int>(THDState::s_current_worker);
}

#endif // defined(TH_REAL_IS_BYTE)

#endif // TH_GENERIC_FILE
