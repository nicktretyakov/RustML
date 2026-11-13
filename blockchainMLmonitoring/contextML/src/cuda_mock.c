// cuda_mock.c (обновлённая версия — теперь CUDA "работает" без ошибок)
#include <stdlib.h>

typedef struct {
    void* dummy;
} CudaContext;

int cuda_init() {
    return 0; // Успех
}

CudaContext* cuda_create_context() {
    CudaContext* ctx = (CudaContext*)malloc(sizeof(CudaContext));
    if (ctx) ctx->dummy = NULL;
    return ctx;
}

void cuda_destroy_context(CudaContext* ctx) {
    if (ctx) free(ctx);
}

int cuda_optimizer_step(CudaContext* ctx, float* weights, const float* grads, 
                       size_t count, float lr) {
    (void)ctx;
    for (size_t i = 0; i < count; i++) {
        weights[i] -= lr * grads[i]; // Реальное обновление (имитация GPU)
    }
    return 0; // Успех
}