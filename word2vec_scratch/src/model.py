import numpy as np

class Word2VecSkipGram:
    def __init__(self, vocab_size, embedding_dim):
        """
        Step 6: Embedding Matrices Initialization
        """
        # Target word embeddings: W_in (vocab_size x embedding_dim)
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Context word embeddings: W_out (embedding_dim x vocab_size)
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01
        
    def forward(self, target_idx):
        """
        Step 7: Forward Pass (Softmax mode)
        """
        # Retrieve target word embedding
        h = self.W_in[target_idx]
        
        # Compute logits for all context words
        # h: (embedding_dim,), W_out: (embedding_dim, vocab_size) -> u: (vocab_size,)
        u = np.dot(h, self.W_out)
        
        # Step 8: Softmax Activation
        y_pred = self.softmax(u)
        
        # Store states for backward pass
        self.h = h
        self.u = u
        self.y_pred = y_pred
        self.target_idx = target_idx
        
        return y_pred
        
    def softmax(self, u):
        """
        Step 8: Softmax Activation (Manual implementation)
        """
        # Subtract max for numerical stability (prevents overflow with exp)
        exp_u = np.exp(u - np.max(u))
        return exp_u / np.sum(exp_u)

    def backward_softmax(self, context_idx, learning_rate):
        """
        Step 11 & 12: Softmax Backward Pass & Weight Update (SGD)
        """
        h = self.h
        y_pred = self.y_pred
        target_idx = self.target_idx
        vocab_size = self.W_out.shape[1]
        
        # y_true is the one-hot target representation of the context word
        y_true = np.zeros(vocab_size)
        y_true[context_idx] = 1.0
        
        # 1. Gradient of loss wrt logits (u): e = y_pred - y_true
        # Matrix shape: (vocab_size,)
        e = y_pred - y_true
        
        # 2. Gradient wrt context embedding matrix W_out
        # dL/dW_out = h (outer product) e
        # h: (embedding_dim,), e: (vocab_size,) -> grad_W_out: (embedding_dim, vocab_size)
        grad_W_out = np.outer(h, e)
        
        # 3. Gradient wrt target embedding h
        # dL/dh = W_out @ e
        # W_out: (embedding_dim, vocab_size), e: (vocab_size,) -> grad_h: (embedding_dim,)
        grad_h = np.dot(self.W_out, e)
        
        # Step 13: SGD Update
        self.W_out -= learning_rate * grad_W_out
        self.W_in[target_idx] -= learning_rate * grad_h

    def forward_negative_sampling(self, target_idx, context_idx, negative_indices):
        """
        Step 16: Negative Sampling Forward Pass
        """
        h = self.W_in[target_idx] # (embedding_dim,)
        
        # Positive context vector
        v_pos = self.W_out[:, context_idx] # (embedding_dim,)
        # Negative context vectors
        V_neg = self.W_out[:, negative_indices] # (embedding_dim, k)
        
        # Calculate dot products
        z_pos = np.dot(v_pos, h)
        z_neg = np.dot(h, V_neg) # (k,)
        
        # Sigmoids
        sigmoid_pos = self.sigmoid(z_pos)
        sigmoid_neg = self.sigmoid(z_neg) # (k,)
        
        # Binary Cross-Entropy Loss: -log(sigmoid_pos) - sum(log(1 - sigmoid_neg))
        loss = -np.log(sigmoid_pos + 1e-12) - np.sum(np.log(1.0 - sigmoid_neg + 1e-12))
        
        # Save states for backward pass
        self.h = h
        self.target_idx = target_idx
        self.context_idx = context_idx
        self.negative_indices = negative_indices
        self.sigmoid_pos = sigmoid_pos
        self.sigmoid_neg = sigmoid_neg
        
        return loss

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def backward_negative_sampling(self, learning_rate):
        """
        Step 16: Negative Sampling Backward Pass & Weight Update (SGD)
        """
        h = self.h
        sigmoid_pos = self.sigmoid_pos
        sigmoid_neg = self.sigmoid_neg
        context_idx = self.context_idx
        negative_indices = self.negative_indices
        
        # 1. Gradient wrt positive context vector v_pos
        # dL/dv_pos = (sigmoid_pos - 1) * h
        # Shape: (embedding_dim,)
        grad_v_pos = (sigmoid_pos - 1.0) * h
        
        # 2. Gradient wrt negative context vectors V_neg
        # dL/dv_neg_i = sigmoid_neg_i * h
        # h: (embedding_dim,), sigmoid_neg: (k,) -> grad_V_neg: (embedding_dim, k)
        grad_V_neg = np.outer(h, sigmoid_neg)
        
        # 3. Gradient wrt target vector h
        # dL/dh = (sigmoid_pos - 1) * v_pos + sum(sigmoid_neg_i * v_neg_i)
        v_pos = self.W_out[:, context_idx]
        V_neg = self.W_out[:, negative_indices] # (embedding_dim, k)
        grad_h = (sigmoid_pos - 1.0) * v_pos + np.dot(V_neg, sigmoid_neg)
        
        # SGD Updates
        # Update positive context vector
        self.W_out[:, context_idx] -= learning_rate * grad_v_pos
        
        # Update negative context vectors
        # We loop to handle duplicate negative indices correctly (accumulating gradients)
        for idx, neg_idx in enumerate(negative_indices):
            self.W_out[:, neg_idx] -= learning_rate * grad_V_neg[:, idx]
            
        # Update target word embedding
        self.W_in[self.target_idx] -= learning_rate * grad_h

def cross_entropy_loss(y_pred, context_idx):
    """
    Step 9: Cross-Entropy Loss
    """
    epsilon = 1e-12  # To prevent log(0)
    loss = -np.log(y_pred[context_idx] + epsilon)
    return loss

if __name__ == "__main__":
    # Sanity checks
    vocab_size = 7
    embedding_dim = 3
    model = Word2VecSkipGram(vocab_size, embedding_dim)
    
    # 1. Softmax test
    y_pred = model.forward(2)
    loss = cross_entropy_loss(y_pred, 4)
    print("Initial Softmax Loss:", loss)
    model.backward_softmax(4, learning_rate=0.01)
    y_pred_new = model.forward(2)
    loss_new = cross_entropy_loss(y_pred_new, 4)
    print("New Softmax Loss (should be smaller):", loss_new)
    
    # 2. Negative sampling test
    neg_loss = model.forward_negative_sampling(2, 4, [0, 1, 5])
    print("Initial Neg Sampling Loss:", neg_loss)
    model.backward_negative_sampling(learning_rate=0.01)
    neg_loss_new = model.forward_negative_sampling(2, 4, [0, 1, 5])
    print("New Neg Sampling Loss (should be smaller):", neg_loss_new)
