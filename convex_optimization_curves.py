import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
import streamlit as st

class ConvexOptimizationCurves:
    """
    Implementation of gradient descent analysis for convex optimization curves
    Based on "Are Convex Optimization Curves Convex?" by Barzilai & Shamir
    """
    
    def __init__(self):
        self.history = []
        
    def gradient_descent(self, f: Callable, grad_f: Callable, x0: float, 
                        eta: float, n_steps: int = 100) -> Tuple[List[float], List[float]]:
        """
        Perform gradient descent on function f with gradient grad_f
        
        Args:
            f: Objective function
            grad_f: Gradient of f
            x0: Initial point
            eta: Step size
            n_steps: Number of iterations
            
        Returns:
            x_history: List of x values
            f_history: List of f(x) values
        """
        x = x0
        x_history = [x]
        f_history = [f(x)]
        
        for _ in range(n_steps):
            x = x - eta * grad_f(x)
            x_history.append(x)
            f_history.append(f(x))
            
        return x_history, f_history
    
    def check_curve_convexity(self, f_history: List[float]) -> bool:
        """
        Check if optimization curve is convex by verifying if 
        differences f(x_n) - f(x_{n+1}) are non-increasing
        """
        differences = [f_history[i] - f_history[i+1] for i in range(len(f_history)-1)]
        
        # Check if differences are non-increasing
        for i in range(len(differences)-1):
            if differences[i] < differences[i+1]:
                return False
        return True
    
    def quadratic_example(self, eta: float, x0: float = 3.0) -> dict:
        """
        Example from the paper: f(x) = x^2
        """
        def f(x):
            return x**2
        
        def grad_f(x):
            return 2*x
        
        x_hist, f_hist = self.gradient_descent(f, grad_f, x0, eta)
        is_convex = self.check_curve_convexity(f_hist)
        
        return {
            'x_history': x_hist,
            'f_history': f_hist,
            'is_convex': is_convex,
            'eta': eta,
            'L': 2.0  # L-smoothness constant for f(x) = x^2
        }
    
    def non_convex_curve_example(self, eta: float = 1.8) -> dict:
        """
        Example showing non-convex optimization curve from Theorem 2
        """
        def f(x):
            if x <= 1:
                return 0.5 * x**2
            else:
                return x - 0.5
        
        def grad_f(x):
            if x <= 1:
                return x
            else:
                return 1
        
        x0 = -1.8
        x_hist, f_hist = self.gradient_descent(f, grad_f, x0, eta, n_steps=5)
        is_convex = self.check_curve_convexity(f_hist)
        
        return {
            'x_history': x_hist,
            'f_history': f_hist,
            'is_convex': is_convex,
            'eta': eta,
            'L': 1.0
        }
    
    def plot_optimization_curve(self, result: dict, title: str = "Optimization Curve"):
        """Plot the optimization curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot function and gradient descent steps
        x_range = np.linspace(min(result['x_history'])-1, max(result['x_history'])+1, 1000)
        
        # For quadratic
        if 'L' in result and result['L'] == 2.0:
            y_range = x_range**2
            ax1.plot(x_range, y_range, 'b-', label='f(x) = x²')
        
        ax1.plot(result['x_history'], result['f_history'], 'ro-', markersize=8, label='GD steps')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'Gradient Descent (η = {result["eta"]:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot optimization curve
        iterations = list(range(len(result['f_history'])))
        ax2.plot(iterations, result['f_history'], 'bo-', markersize=6)
        ax2.set_xlabel('Iteration n')
        ax2.set_ylabel('f(x_n)')
        ax2.set_title(f'{title} - Convex: {result["is_convex"]}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class OptimizationChatbot:
    """Chatbot for answering questions about convex optimization curves"""
    
    def __init__(self):
        self.analyzer = ConvexOptimizationCurves()
        self.knowledge_base = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> dict:
        """Build knowledge base from the paper"""
        return {
            'main_results': {
                'theorem_1': "For L-smooth convex functions with step size η ∈ (0, 1/L], the optimization curve is convex.",
                'theorem_2': "For step sizes η ∈ (1.75/L, 2/L), the optimization curve may not be convex despite monotonic convergence.",
                'theorem_3': "For η ∈ (0, 2/L], the gradient norm sequence {||∇f(x_n)||} is non-increasing."
            },
            'key_concepts': {
                'optimization_curve': "The linear interpolation of points {(n, f(x_n))} in R²",
                'convex_curve': "A curve where differences f(x_n) - f(x_{n+1}) are non-increasing",
                'L_smooth': "A function with L-Lipschitz gradient: ||∇f(x) - ∇f(y)|| ≤ L||x - y||"
            },
            'step_size_regimes': {
                'optimal': "η ∈ (0, 1/L] guarantees both convergence and convex optimization curve",
                'convergent_non_convex': "η ∈ (1.75/L, 2/L) ensures convergence but may have non-convex curve",
                'unknown': "η ∈ (1/L, 1.75/L] - behavior not fully characterized in the paper"
            }
        }
    
    def answer_question(self, question: str) -> str:
        """Answer questions about convex optimization curves"""
        question_lower = question.lower()
        
        # Check for specific topics
        if "convex" in question_lower and "curve" in question_lower:
            return self._explain_convex_curves()
        elif "step size" in question_lower or "eta" in question_lower:
            return self._explain_step_sizes()
        elif "theorem" in question_lower:
            return self._explain_theorems()
        elif "gradient norm" in question_lower:
            return self._explain_gradient_norm()
        elif "example" in question_lower:
            return self._provide_example()
        else:
            return self._general_response()
    
    def _explain_convex_curves(self) -> str:
        return """**Convex Optimization Curves:**

An optimization curve is the path traced by f(x_n) over iterations n. The key finding is:

• For L-smooth convex functions, the curve is **convex** when η ≤ 1/L
• This means f(x_n) - f(x_{n+1}) decreases monotonically
• Convexity prevents undesirable plateaus followed by sharp drops

The curve being convex is stronger than just monotonic decrease - it ensures smooth, predictable convergence."""

    def _explain_step_sizes(self) -> str:
        return """**Step Size Regimes for L-smooth Convex Functions:**

1. **η ∈ (0, 1/L]**: Optimal regime
   - Monotonic convergence ✓
   - Convex optimization curve ✓
   - Includes worst-case optimal η = 1/L

2. **η ∈ (1.75/L, 2/L)**: Problematic regime
   - Monotonic convergence ✓
   - Convex optimization curve ✗
   - Can have non-convex curves despite convergence

3. **η ∈ (1/L, 1.75/L]**: Open question
   - Behavior not fully characterized

4. **η > 2/L**: Divergence
   - Gradient descent may diverge"""

    def _explain_theorems(self) -> str:
        return """**Main Theorems:**

**Theorem 1**: For convex L-smooth functions with η ∈ (0, 1/L], the optimization curve is convex.

**Theorem 2**: There exist convex L-smooth functions where η ∈ (1.75/L, 2/L) produces non-convex curves.

**Theorem 3**: For convex L-smooth functions with η ∈ (0, 2/L], gradient norms ||∇f(x_n)|| decrease monotonically.

**Key Insight**: There's a surprising gap between monotonic convergence (η < 2/L) and convex curves (η ≤ 1/L)."""

    def _explain_gradient_norm(self) -> str:
        return """**Gradient Norm Behavior:**

For convex L-smooth functions with η ∈ (0, 2/L]:
• The sequence {||∇f(x_n)||} is **monotonically decreasing**
• This holds for the entire convergence regime
• Different from optimization curve convexity!

This means:
- Gradients consistently get smaller
- No oscillations in gradient magnitude
- Applies even when the optimization curve is non-convex (e.g., η ∈ (1.75/L, 2/L))"""

    def _provide_example(self) -> str:
        return """**Example: f(x) = x²**

• L-smoothness constant: L = 2
• Optimal step size range: η ∈ (0, 0.5]
• With η = 0.1, x₀ = 3:
  - x₁ = 2.4, x₂ = 1.92, ...
  - Optimization curve is convex
  
**Counter-example** (Theorem 2):
• Piecewise function: f(x) = {x²/2 if x≤1, x-0.5 if x>1}
• With η = 1.8, x₀ = -1.8:
  - Non-convex optimization curve
  - f(x₀) - f(x₁) < f(x₁) - f(x₂)"""

    def _general_response(self) -> str:
        return """**Convex Optimization Curves - Key Points:**

This research addresses whether gradient descent produces convex optimization curves on convex functions.

**Main Findings:**
• Step size critically affects curve convexity
• η ≤ 1/L guarantees convex curves
• η ∈ (1.75/L, 2/L) can produce non-convex curves
• Gradient norms always decrease for η ≤ 2/L

**Practical Implications:**
- Use η ≤ 1/L to avoid misleading plateaus
- Larger step sizes may converge but behave unpredictably

Would you like details on any specific aspect?"""

def create_streamlit_app():
    """Create Streamlit app for interactive exploration"""
    st.set_page_config(page_title="Convex Optimization Curves", layout="wide")
    
    st.title("Are Convex Optimization Curves Convex?")
    st.markdown("*Based on the paper by Guy Barzilai and Ohad Shamir*")
    
    # Initialize components
    analyzer = ConvexOptimizationCurves()
    chatbot = OptimizationChatbot()
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio("Select Mode:", ["Interactive Q&A", "Visualizations", "Theory"])
    
    if mode == "Interactive Q&A":
        st.header("Ask Questions About Convex Optimization Curves")
        
        # Initialize session state for chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about convex optimization curves..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            response = chatbot.answer_question(prompt)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    
    elif mode == "Visualizations":
        st.header("Visualize Optimization Curves")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quadratic Function Example")
            eta_quad = st.slider("Step size η", 0.01, 1.0, 0.1, 0.01, key="quad")
            x0_quad = st.slider("Initial point x₀", -5.0, 5.0, 3.0, 0.1, key="x0")
            
            if st.button("Run Gradient Descent", key="run_quad"):
                result = analyzer.quadratic_example(eta_quad, x0_quad)
                fig = analyzer.plot_optimization_curve(result, "f(x) = x²")
                st.pyplot(fig)
                
                st.info(f"L = 2.0, η/L = {eta_quad/2:.3f}")
                if result['is_convex']:
                    st.success("✓ Optimization curve is convex")
                else:
                    st.warning("✗ Optimization curve is NOT convex")
        
        with col2:
            st.subheader("Non-Convex Curve Example")
            eta_non = st.slider("Step size η", 1.76, 1.99, 1.8, 0.01, key="non")
            
            if st.button("Run Example", key="run_non"):
                result = analyzer.non_convex_curve_example(eta_non)
                
                st.write("**Function values:**")
                for i, (x, f) in enumerate(zip(result['x_history'][:3], 
                                              result['f_history'][:3])):
                    st.write(f"x_{i} = {x:.3f}, f(x_{i}) = {f:.3f}")
                
                st.write("\n**Differences:**")
                st.write(f"f(x₀) - f(x₁) = {result['f_history'][0] - result['f_history'][1]:.3f}")
                st.write(f"f(x₁) - f(x₂) = {result['f_history'][1] - result['f_history'][2]:.3f}")
                
                if result['is_convex']:
                    st.success("✓ Optimization curve is convex")
                else:
                    st.error("✗ Optimization curve is NOT convex (as expected!)")
    
    elif mode == "Theory":
        st.header("Theoretical Results")
        
        tab1, tab2, tab3 = st.tabs(["Main Theorems", "Key Concepts", "Implications"])
        
        with tab1:
            st.markdown("""
            ### Theorem 1 (Convex Optimization Curves)
            For convex L-smooth functions with step size **η ∈ (0, 1/L]**, 
            the optimization curve is **convex**.
            
            ### Theorem 2 (Non-Convex Curves Exist)
            For every L > 0, there exists a convex L-smooth function such that 
            for **η ∈ (1.75/L, 2/L)**, the optimization curve is **not convex**.
            
            ### Theorem 3 (Gradient Norm Monotonicity)
            For convex L-smooth functions with **η ∈ (0, 2/L]**, 
            the gradient norm sequence {||∇f(xₙ)||} is **non-increasing**.
            """)
        
        with tab2:
            st.markdown("""
            ### Optimization Curve
            The linear interpolation of points {(n, f(xₙ))} in ℝ².
            
            ### Convex Curve Property
            The sequence {f(xₙ) - f(xₙ₊₁)} is non-increasing.
            
            ### L-Smooth Function
            A differentiable function where:
            ||∇f(x) - ∇f(y)|| ≤ L||x - y|| for all x, y
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Guidelines
            
            1. **Use η ≤ 1/L** for predictable, smooth convergence
            2. **Avoid η > 1.75/L** to prevent non-convex behavior
            3. **η = 1/L** is worst-case optimal and guarantees convexity
            
            ### Why This Matters
            
            - **Avoiding premature stopping**: Non-convex curves can plateau then drop
            - **Algorithm tuning**: Choose step sizes for desired behavior
            - **Understanding convergence**: Monotonic ≠ convex curve
            """)

if __name__ == "__main__":
    # For testing individual components
    analyzer = ConvexOptimizationCurves()
    chatbot = OptimizationChatbot()
    
    # Test quadratic example
    result = analyzer.quadratic_example(eta=0.1)
    print(f"Quadratic example - Convex curve: {result['is_convex']}")
    
    # Test chatbot
    print("\nChatbot test:")
    print(chatbot.answer_question("What are convex optimization curves?"))