#include <stdio.h>
#include <math.h>
#include <time.h>

typedef struct {
    float cpu;    // x1
    float ram;    // x2
    float Target; // y
} Data;

typedef struct {
    float w1;
    float w2;
    float b;
} Neuron;

Data td[] = {
    {0.20f, 0.30f, 0.0f},
    {0.35f, 0.40f, 0.0f},
    {0.55f, 0.35f, 0.0f},
    {0.40f, 0.60f, 0.0f},
    {0.65f, 0.55f, 0.0f},
    {0.70f, 0.60f, 1.0f},
    {0.80f, 0.45f, 1.0f},
    {0.55f, 0.85f, 1.0f},
    {0.90f, 0.70f, 1.0f},
    {0.30f, 0.90f, 0.0f},
    {0.75f, 0.70f, 1.0f},
    {0.60f, 0.60f, 0.0f},
    {0.50f, 0.70f, 0.0f},
    {0.80f, 0.80f, 1.0f},
    {0.70f, 0.50f, 0.0f},
    {0.60f, 0.75f, 1.0f},
}; // Source: ChatGPT

#define TRAIN_COUNT (sizeof(td) / sizeof(td[0]))

float rand_float(void) { 
  return (float)rand() / (float)RAND_MAX; 
}

// ─────────────────────────────────────────────
// Sigmoid function
//
//             1
//   σ(z) = ─────────
//           1 + e⁻ᶻ
// ─────────────────────────────────────────────
float sigmoid(float z)
{
    return (1.0f) / (1.0f + expf(-z));
}

// ─────────────────────────────────────────────
// Forward Pass
//
//   z = x1*w1 + x2*w2 + b
//   a = σ(z)
// ─────────────────────────────────────────────
float predict(Neuron n, float cpu, float ram)
{
    float z = n.w1 * cpu + n.w2 * ram + n.b;
    return sigmoid(z);
}

// ─────────────────────────────────────────────
// Cost function
//
//             1    N
//   C(w1,w2,b)= ─── · Σ (aᵢ - yᵢ)²
//             2N  i=1
//
//   aᵢ = σ(x1ᵢ*w1 + x2ᵢ*w2 + b)
// ─────────────────────────────────────────────
float cost(Neuron n)
{
    float loss = 0.0f;
    for(size_t i = 0; i < TRAIN_COUNT; ++i) {
        float a = predict(n, td[i].cpu, td[i].ram);
        float d = a - td[i].Target;

        loss += 0.5f * d * d;
    }

    return loss / TRAIN_COUNT;
}

// ─────────────────────────────────────────────
// Gradient of cost w.r.t. weight
//
//   δᵢ = (aᵢ - yᵢ) * aᵢ * (1 - aᵢ)
//
//   ∂C/∂w1 = (1/N) * Σ δᵢ * x1ᵢ
//   ∂C/∂w2 = (1/N) * Σ δᵢ * x2ᵢ
// ─────────────────────────────────────────────
float dw(Neuron n, int input)
{
    float gradient = 0.0f;

    for(size_t i = 0; i < TRAIN_COUNT; ++i) {
        float cpu = td[i].cpu;
        float ram = td[i].ram;
        float target = td[i].Target;

        float a = predict(n, cpu, ram);
        float d = a - target;
        float delta = d * a * (1.0f - a);

        if (input == 1)
            gradient += delta * cpu;
        else if (input == 2)
            gradient += delta * ram;
    }

    return gradient / TRAIN_COUNT;
}

// ─────────────────────────────────────────────
// Gradient of cost w.r.t. bias b
//
//   ∂C/∂b = (1/N) * Σ δᵢ
// ─────────────────────────────────────────────
float db(Neuron n)
{
    float gradient = 0.0f;

    for(size_t i = 0; i < TRAIN_COUNT; ++i) {
        float a = predict(n, td[i].cpu, td[i].ram);
        float d = a - td[i].Target;
        float delta = d * a * (1.0f - a);
        gradient += delta;
    }

    return gradient / TRAIN_COUNT;
}

// ─────────────────────────────────────────────
// Gradient Descent
//
//   w1 = w1 - rate * ∂C/∂w1
//   w2 = w2 - rate * ∂C/∂w2
//    b =  b - rate * ∂C/∂b
// ─────────────────────────────────────────────
void train(Neuron *n, float rate, float iterations)
{
    for(size_t i = 0; i < iterations; ++i) {
        float gradient_cpu = dw(*n, 1);
        float gradient_ram = dw(*n, 2);
        float gradient_target = db(*n);

        n->w1 -= rate * gradient_cpu;
        n->w2 -= rate * gradient_ram;
        n->b -= rate * gradient_target;
    }
}

int main(void)
{
    Neuron n = {0.0f, 0.0f, 0.0f};

    printf("Initial cost: %.6f\n", cost(n));
    train(&n, 1.0f, 100000);
    printf("Final cost: %.6f\n\n", cost(n));

    puts("0 -> low");
    puts("1 -> high");

    while(1) {
        float cpu_input = 0.0f;
        float ram_input = 0.0f;

        puts("Enter CPU & RAM value (0-100):");
        printf("cpu: ");
        scanf("%f", &cpu_input);
        printf("ram: ");
        scanf("%f", &ram_input);

        float scaled_cpu = cpu_input / 100.0f;
        float scaled_ram = ram_input / 100.0f;

        float output = predict(n, scaled_cpu, scaled_ram);

        printf("Output probability: %.4f\n", output);
        printf("Prediction: %.0f\n\n", output);
    }

    return 0;
}
