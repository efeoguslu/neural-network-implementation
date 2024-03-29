#include "matrix.h"
#include "helpers.h"
#include "activation.h"
#include "xor.h"

// 2 inputs, 2 hidden layers with 4 and 2 neurons respectively, and one output layer with 1 neuron
// size_t arch[] = {2, 4, 2, 1};
// Network nn = nn_alloc(arch, ARRAY_LEN(layers));

Network nn_alloc(size_t *arch, size_t arch_count){
    MATRIX_ASSERT(arch_count > 0);
    Network nn;
    nn.count = arch_count - 1; // subtracting the input layer
    // preallocating arrays of matrices
    nn.ws = MATRIX_MALLOC(sizeof(*nn.ws)*nn.count);
    MATRIX_ASSERT(nn.ws != NULL);
    nn.bs = MATRIX_MALLOC(sizeof(*nn.bs)*nn.count);
    MATRIX_ASSERT(nn.bs != NULL);
    nn.as = MATRIX_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    MATRIX_ASSERT(nn.as != NULL);

    // input vector with a single row:
    nn.as[0] = mat_alloc(1, arch[0]);

    for(size_t i = 1; i < arch_count; ++i){
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i]     = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(Network nn, const char* name){
    
    char buf[256];
    printf("%s = [\n", name);

    for(size_t i = 0; i < nn.count; ++i){
        snprintf(buf, sizeof(buf), "ws%lu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%lu", i);
        mat_print(nn.bs[i], buf, 4);
    }

    printf("]\n");
}

void nn_rand(Network nn, double low, double high){
    for(size_t i = 0; i < nn.count; ++i){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(Network nn){
    for(size_t i = 0; i < nn.count; ++i){
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

double nn_cost(Network nn, Mat ti, Mat to){
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;
    double c = 0;
    for(size_t i = 0; i < n; ++i){
        Mat x = mat_row(ti, i); // expected input
        Mat y = mat_row(to, i); // expected output
        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;
        for(size_t j = 0; j < q; ++j){
            double d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }
    return c/n;
}


void nn_finite_diff(Network nn, Network g, double eps, Mat ti, Mat to){

    double saved; 
    double c = nn_cost(nn, ti, to);

    for(size_t i = 0; i < nn.count; ++i){

        for(size_t j = 0; j < nn.ws[i].rows; ++j){
            for(size_t k = 0; k < nn.ws[i].cols; ++k){
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for(size_t j = 0; j < nn.bs[i].rows; ++j){
            for(size_t k = 0; k < nn.bs[i].cols; ++k){
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }

    }
}

void nn_learn(Network nn, Network g, double rate){
    for(size_t i = 0; i < nn.count; ++i){
        for(size_t j = 0; j < nn.ws[i].rows; ++j){
            for(size_t k = 0; k < nn.ws[i].cols; ++k){
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
            }
        }
        for(size_t j = 0; j < nn.bs[i].rows; ++j){
            for(size_t k = 0; k < nn.bs[i].cols; ++k){
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);

            }
        }
    }
}

void nn_backprop(Network nn, Network g, Mat ti, Mat to){
    MATRIX_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    MATRIX_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);
    // i     --> current sample
    // layer --> current layer
    // j     --> current activation
    // k     --> previous activation

    for(size_t i = 0; i < n; ++i){
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for(size_t j = 0; j <= nn.count; ++j){
            mat_fill(g.as[j], 0);
        }

        for(size_t j = 0; j < to.cols; ++j){
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        for(size_t layer = nn.count; layer > 0; --layer){
            for(size_t j = 0; j < nn.as[layer].cols; ++j){
                double a = MAT_AT(nn.as[layer], 0, j);
                double da = MAT_AT(g.as[layer], 0, j);
                MAT_AT(g.bs[layer - 1], 0, j) += 2*da*a*(1 - a); // partial derivative for biases

                for(size_t k = 0; k < nn.as[layer - 1].cols; ++k){
                    // j --> weight matrix column
                    // k --> weight matrix row
                    double pa = MAT_AT(nn.as[layer - 1], 0, k);
                    double w  = MAT_AT(nn.ws[layer - 1], k, j); 
                    MAT_AT(g.ws[layer - 1], k, j) += 2*da*a*(1 - a)*pa; // partial derivative for weights 
                    MAT_AT(g.as[layer - 1], 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }
    }

    for(size_t i = 0; i < g.count; ++i){

        for(size_t j = 0; j < g.ws[i].rows; ++j){
            for(size_t k = 0; k < g.ws[i].cols; ++k){
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }

        for(size_t j = 0; j < g.bs[i].rows; ++j){
            for(size_t k = 0; k < g.bs[i].cols; ++k){
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}

void nn_zero(Network nn){
    for(size_t i = 0; i < nn.count; ++i){
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
}


void mat_fill(Mat m, double x){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = x;
        }
    }
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = MATRIX_CALLOC(rows*cols, sizeof(*m.es)); 
    MATRIX_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dst, Mat a, Mat b){
    // checking if matrices are eligible for dot product:
    MATRIX_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == b.cols);
    // dot product operation:
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < n; ++k){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat mat_row(Mat m, size_t row){
    return (Mat){ 
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dst, Mat src){
    MATRIX_ASSERT(dst.rows == src.rows);
    MATRIX_ASSERT(dst.cols == src.cols);

    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Mat dst, Mat a){
    // checking if matrices are eligible to summation:
    MATRIX_ASSERT(dst.rows == a.rows);
    MATRIX_ASSERT(dst.cols == a.cols);
    // the summation:
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_print(Mat m, const char* name, size_t padding){

    // printf("\nMatrix with %u rows and %u columns:\n", m.rows, m.cols);
    
    printf("%*s%s = [\n", (int)padding, "", name);
    for(size_t i = 0; i < m.rows; ++i){
        printf("%*s     ", (int)padding, "");
        for(size_t j = 0; j < m.cols; ++j){
            printf("%lf ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
    
}

void mat_rand(Mat m, double low, double high){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = rand_double()*(high - low) + low;
        }
        
    }
}

void mat_sig(Mat m){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = sigmoid(MAT_AT(m, i, j));
        }
    }
}

void mat_save(FILE *file, Mat m) {
    fprintf(file, "%zu %zu\n", m.rows, m.cols);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            fprintf(file, "%lf ", MAT_AT(m, i, j));
        }
        fprintf(file, "\n");
    }
}

void mat_free(Mat m) {
    free(m.es);
}

void mat_shuffle_rows(Mat m){
    for(size_t i = 0; i < m.rows; ++i){
        size_t j = i + rand()%(m.rows - i);
        if(i != j){
            for(size_t k = 0; k < m.cols; ++k){
                double t = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = t;
            }
        }
    }
}


// --------------------------------------------------------------------------------


/*

Forward propagation: The network processes the input data to produce an output.
Loss computation: The difference between the network's output and the target output is calculated.
Backward propagation: The error is propagated backward through the network, computing gradients for each layer's weights and biases.
Gradient averaging: The gradients are averaged over all samples to adjust the network parameters in the training process.
This code assumes a specific activation function (possibly sigmoid, given the a * (1 - a) term in the gradient calculations) and loss function (possibly mean squared error, given the 2 * da term). Ensure these assumptions are correct for your network before using the code.
*/



void nn_backpropagation(Network nn, Network g, Mat ti, Mat to){
    // Ensure the input and target output matrices have the same number of rows
    MATRIX_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows; // Number of training samples
    // Ensure the network's output layer and the target output matrix have the same number of columns
    MATRIX_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g); // Initialize gradient network to zero

    // Loop over each training sample
    for(size_t i = 0; i < n; ++i){
        // Set the current input sample to the network's input layer
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        // Perform a forward pass to compute the network's output
        nn_forward(nn);

        // Initialize gradients for each layer to zero
        for(size_t j = 0; j <= nn.count; ++j){
            mat_fill(g.as[j], 0);
        }

        // Calculate the output layer's error (gradient of the loss w.r.t. the output)
        for(size_t j = 0; j < to.cols; ++j){
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        // Propagate the error backwards through the network
        for(size_t layer = nn.count; layer > 0; --layer){
            for(size_t j = 0; j < nn.as[layer].cols; ++j){
                double a = MAT_AT(nn.as[layer], 0, j); // Activation of current neuron
                double da = MAT_AT(g.as[layer], 0, j); // Gradient of the loss w.r.t. current neuron's activation
                // Compute gradient for biases and update
                MAT_AT(g.bs[layer - 1], 0, j) += 2*da*a*(1 - a); // Assuming sigmoid activation function for simplicity

                // Loop over all connections coming into current neuron
                for(size_t k = 0; k < nn.as[layer - 1].cols; ++k){
                    double pa = MAT_AT(nn.as[layer - 1], 0, k); // Activation of previous neuron
                    double w  = MAT_AT(nn.ws[layer - 1], k, j); // Weight of connection from previous neuron
                    // Compute gradient for weights and update
                    MAT_AT(g.ws[layer - 1], k, j) += 2*da*a*(1 - a)*pa; // Gradient for weight
                    // Accumulate gradient for activations of neurons in the previous layer (for backpropagation)
                    MAT_AT(g.as[layer - 1], 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }
    }

    // Average the gradients over all samples
    for(size_t i = 0; i < g.count; ++i){
        // Average gradients for weights
        for(size_t j = 0; j < g.ws[i].rows; ++j){
            for(size_t k = 0; k < g.ws[i].cols; ++k){
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }

        // Average gradients for biases
        for(size_t j = 0; j < g.bs[i].rows; ++j){
            for(size_t k = 0; k < g.bs[i].cols; ++k){
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}
