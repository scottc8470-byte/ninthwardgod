import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Dict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web

# Keep existing ConvexOptimizationCurves class
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
        if len(f_history) < 3:
            return True
            
        differences = [f_history[i] - f_history[i+1] for i in range(len(f_history)-1)]
        
        # Check if differences are non-increasing
        for i in range(len(differences)-1):
            if differences[i] < differences[i+1] - 1e-10:  # Small tolerance for numerical errors
                return False
        return True
    
    def quadratic_example(self, eta: float, x0: float = 3.0, n_steps: int = 50) -> dict:
        """
        Example from the paper: f(x) = x^2
        """
        def f(x):
            return x**2
        
        def grad_f(x):
            return 2*x
        
        x_hist, f_hist = self.gradient_descent(f, grad_f, x0, eta, n_steps)
        is_convex = self.check_curve_convexity(f_hist)
        
        return {
            'x_history': x_hist,
            'f_history': f_hist,
            'is_convex': is_convex,
            'eta': eta,
            'L': 2.0  # L-smoothness constant for f(x) = x^2
        }
    
    def plot_optimization_curve(self, result: dict, title: str = "Optimization Curve"):
        """Plot the optimization curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot function and gradient descent steps
        x_range = np.linspace(min(result['x_history'])-1, max(result['x_history'])+1, 1000)
        
        # For quadratic
        if 'L' in result and result['L'] == 2.0:
            y_range = x_range**2
            ax1.plot(x_range, y_range, 'b-', label='f(x) = x²', alpha=0.7)
        
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

class ComprehensiveChatbot:
    """Enhanced chatbot for all topics: Convex Optimization, AI Agents, and Transformer Architectures"""
    
    def __init__(self):
        self.analyzer = ConvexOptimizationCurves()
        self.knowledge_base = self._build_comprehensive_knowledge_base()
    
    def _build_comprehensive_knowledge_base(self) -> dict:
        """Build comprehensive knowledge base for all topics"""
        return {
            'convex_optimization': {
                'main_results': {
                    'theorem_1': "For L-smooth convex functions with step size η ∈ (0, 1/L], the optimization curve is convex.",
                    'theorem_2': "For step sizes η ∈ (1.75/L, 2/L), the optimization curve may not be convex despite monotonic convergence.",
                    'theorem_3': "For η ∈ (0, 2/L], the gradient norm sequence {||∇f(x_n)||} is non-increasing."
                }
            },
            'ai_agents': {
                'python_skills': {
                    'FastAPI': "Build lightweight, production-ready APIs",
                    'Async Programming': "Handle multiple tasks asynchronously",
                    'Pydantic': "Data validation and settings management",
                    'Logging': "Essential for debugging agents",
                    'Testing': "Unit and integration tests for complex agents",
                    'Database Management': "SQLAlchemy + Alembic for RAG and memories"
                },
                'rag_skills': {
                    'Understanding RAG': "Retrieval-Augmented Generation fundamentals",
                    'Text Embeddings': "Foundation of search and retrieval",
                    'Vector Database': "Store and retrieve embeddings efficiently",
                    'Chunking Strategies': "Split data smartly for better retrieval",
                    'RAG with PostgreSQL': "RAG without expensive vector DBs",
                    'RAG with LangChain': "Chain together retrieval and LLMs",
                    'RAG Evaluations': "Measure retrieval and answer quality",
                    'Advanced RAG': "Fine-tune for real-world performance"
                },
                'frameworks': {
                    'LangGraph': "Production-ready AI agent framework"
                },
                'monitoring': {
                    'LangFuse': "AI agent monitoring and observability"
                }
            },
            'transformer_architectures': {
                'deepseek_v3': {
                    'size': "671B total, 37B active parameters",
                    'key_features': ["Multi-Head Latent Attention (MLA)", "Mixture-of-Experts (256 experts, 9 active)", "Shared expert"],
                    'innovations': "MLA compresses KV cache, outperforms GQA"
                },
                'llama_4': {
                    'size': "400B total, 17B active parameters",
                    'key_features': ["Grouped-Query Attention", "MoE (2 active experts)", "Alternating MoE/dense layers"],
                    'innovations': "Classic MoE setup with fewer but larger experts"
                },
                'gemma_3': {
                    'size': "27B parameters",
                    'key_features': ["Sliding window attention (1024 tokens)", "5:1 local:global ratio", "Pre+Post normalization"],
                    'innovations': "Efficient local attention reduces KV cache by 40%"
                },
                'qwen3': {
                    'sizes': "0.6B to 235B (dense and MoE)",
                    'key_features': ["Dense: deeper architecture", "MoE: no shared expert", "QK-norm for stability"],
                    'innovations': "Excellent performance across all size classes"
                },
                'kimi_k2': {
                    'size': "1 trillion parameters",
                    'key_features': ["Based on DeepSeek-V3 architecture", "More experts (380 vs 256)", "Muon optimizer"],
                    'innovations': "First production use of Muon optimizer at scale"
                },
                'trends_2025': {
                    'moe_adoption': "Widespread adoption of Mixture-of-Experts",
                    'attention_efficiency': "MLA vs GQA vs sliding window approaches",
                    'normalization': "Various Pre/Post-Norm configurations",
                    'architecture_similarity': "Core transformer design remains stable"
                }
            }
        }
    
    def answer_question(self, question: str) -> str:
        """Answer questions about any of the three topics"""
        question_lower = question.lower()
        
        # Convex Optimization questions
        if any(term in question_lower for term in ["convex", "optimization curve", "gradient descent", "step size"]):
            return self._answer_convex_optimization(question_lower)
        
        # AI Agent questions
        elif any(term in question_lower for term in ["ai agent", "rag", "fastapi", "langchain", "vector", "embedding"]):
            return self._answer_ai_agents(question_lower)
        
        # Transformer Architecture questions
        elif any(term in question_lower for term in ["transformer", "deepseek", "llama", "gemma", "qwen", "kimi", "moe", "attention"]):
            return self._answer_transformers(question_lower)
        
        # General questions
        elif "what is this" in question_lower or "about" in question_lower:
            return self._about_site()
        
        else:
            return self._general_response()
    
    def _about_site(self) -> str:
        return """**Welcome to Ninth Ward God - AI Knowledge Hub!**

This comprehensive platform covers three cutting-edge AI topics:

🎯 **1. Convex Optimization Curves**
Explore the fascinating question: "Are Convex Optimization Curves Convex?" Based on Barzilai & Shamir's 2025 research.

🤖 **2. AI Agent Development**
Learn critical skills for building production AI agents, including Python tools, RAG systems, and frameworks.

🏗️ **3. Modern Transformer Architectures**
Deep dive into 2025's state-of-the-art LLM architectures: DeepSeek V3, Llama 4, Gemma 3, and more.

Ask me anything about these topics or explore the interactive visualizations!"""

    def _answer_convex_optimization(self, question: str) -> str:
        """Handle convex optimization questions"""
        if "theorem" in question:
            return """**Convex Optimization Theorems:**

**Theorem 1** 📐: For convex L-smooth functions with η ∈ (0, 1/L], the optimization curve is convex.

**Theorem 2** 🔄: There exist convex L-smooth functions where η ∈ (1.75/L, 2/L) produces non-convex curves.

**Theorem 3** 📉: For convex L-smooth functions with η ∈ (0, 2/L], gradient norms ||∇f(x_n)|| decrease monotonically.

**Key Insight** 💡: There's a gap between monotonic convergence (η < 2/L) and convex curves (η ≤ 1/L)."""
        
        elif "step size" in question:
            return """**Step Size Regimes for L-smooth Convex Functions:**

1. **η ∈ (0, 1/L]**: ✅ Optimal regime
   - Monotonic convergence ✓
   - Convex optimization curve ✓
   
2. **η ∈ (1.75/L, 2/L)**: ⚠️ Problematic regime
   - Monotonic convergence ✓
   - Convex optimization curve ✗
   
3. **η > 2/L**: ❌ Divergence
   - Gradient descent may diverge"""
        
        else:
            return """**Convex Optimization Curves:**

An optimization curve traces f(x_n) over iterations. Key findings:
• Curve is **convex** when η ≤ 1/L for L-smooth functions
• Convexity prevents misleading plateaus
• Stronger property than just monotonic decrease"""

    def _answer_ai_agents(self, question: str) -> str:
        """Handle AI agent development questions"""
        if "rag" in question:
            return """**RAG (Retrieval-Augmented Generation) Skills:**

📚 **Core Components:**
• **Text Embeddings**: Convert text to vectors for similarity search
• **Vector Database**: Store and retrieve embeddings (Pinecone, Weaviate, pgvector)
• **Chunking**: Split documents intelligently (by tokens, sentences, or semantic units)

🛠️ **Implementation Path:**
1. Start with LangChain for quick prototyping
2. Use PostgreSQL + pgvector for cost-effective production
3. Implement proper evaluation metrics (relevance, accuracy)
4. Optimize with advanced techniques (HyDE, multi-query)

📊 **Key Resources:**
• Understanding RAG: https://lnkd.in/dGUijEMw
• RAG with PostgreSQL: https://lnkd.in/dDm7miwh
• RAG Evaluations: https://lnkd.in/dn-NDF_U"""
        
        elif "python" in question or "skills" in question:
            return """**Essential Python Skills for AI Agents:**

🚀 **Core Development Skills:**
• **FastAPI**: Build async APIs for agent endpoints
• **Pydantic**: Validate LLM inputs/outputs
• **Async Programming**: Handle concurrent agent tasks
• **SQLAlchemy + Alembic**: Manage agent memory/state

🧪 **Production Essentials:**
• **Logging**: Track agent decisions and errors
• **Testing**: Unit tests for components, integration tests for workflows
• **Error Handling**: Graceful fallbacks for LLM failures

📦 **Framework:**
• **LangGraph**: Production-ready agent orchestration
• **LangFuse**: Monitor agent performance

Start with a simple FastAPI + LangChain project and gradually add complexity!"""
        
        else:
            return """**AI Agent Development Overview:**

Building production AI agents requires:

🎯 **Python Foundation**: FastAPI, async programming, Pydantic
🔍 **RAG Systems**: Embeddings, vector DBs, retrieval strategies  
🏗️ **Frameworks**: LangGraph for orchestration
📊 **Monitoring**: LangFuse for observability

Start with hands-on projects and learn each component as needed!"""

    def _answer_transformers(self, question: str) -> str:
        """Handle transformer architecture questions"""
        if "deepseek" in question:
            return """**DeepSeek V3/R1 Architecture:**

📏 **Scale**: 671B total parameters, 37B active

🔧 **Key Innovations:**
• **Multi-Head Latent Attention (MLA)**: Compresses KV cache, outperforms GQA
• **Mixture-of-Experts**: 256 experts, 9 active (1 shared + 8 selected)
• **Shared Expert**: Always active, learns common patterns

💡 **Why It Matters:**
- MLA reduces memory while improving performance
- MoE enables massive scale with efficient inference
- Foundation for reasoning model DeepSeek R1"""
        
        elif "comparison" in question or "trends" in question:
            return """**2025 Transformer Architecture Trends:**

🏗️ **Common Patterns:**
• Core transformer design remains stable (7 years after GPT!)
• RoPE for positions, SwiGLU activations, RMSNorm

📊 **Key Differentiators:**

**1. Attention Mechanisms:**
- DeepSeek/Kimi: Multi-Head Latent Attention (MLA)
- Llama/Qwen: Grouped-Query Attention (GQA)  
- Gemma: Sliding window attention

**2. Scale Strategies:**
- MoE adoption: DeepSeek, Llama 4, Qwen3, Kimi K2
- Dense models: Gemma 3, smaller Qwen3 variants

**3. Optimizations:**
- Normalization placement variations
- QK-Norm for stability (OLMo 2, Gemma)
- Shared vs no shared experts in MoE

The race is on efficiency, not radical redesigns!"""
        
        elif any(model in question for model in ["llama", "gemma", "qwen", "kimi"]):
            model_info = {
                "llama": """**Llama 4 (400B, 17B active):**
• Classic MoE with 2 active experts
• Alternates MoE and dense layers
• Uses GQA instead of MLA""",
                
                "gemma": """**Gemma 3 (27B):**
• Sliding window attention (1024 tokens)
• 5:1 local:global attention ratio
• Both Pre and Post normalization
• 40% KV cache reduction""",
                
                "qwen": """**Qwen3 Family (0.6B to 235B):**
• Dense models: deeper architectures
• MoE models: no shared expert
• QK-Norm for training stability
• Excellent performance/size ratio""",
                
                "kimi": """**Kimi K2 (1 trillion parameters):**
• Largest open-weight model
• Based on DeepSeek V3 architecture
• First production use of Muon optimizer
• 380 experts vs DeepSeek's 256"""
            }
            
            for model, info in model_info.items():
                if model in question:
                    return info
            
        return """**Modern Transformer Architectures (2025):**

Major models: DeepSeek V3, Llama 4, Gemma 3, Qwen3, Kimi K2

Key trends:
• Widespread MoE adoption for scale
• Various attention efficiency approaches
• Core design stability with incremental improvements

Ask about specific models for detailed comparisons!"""

    def _general_response(self) -> str:
        return """**Ninth Ward God - AI Knowledge Hub**

Choose a topic to explore:

🎯 **Convex Optimization**: Step sizes, theorems, curve convexity
🤖 **AI Agents**: Python skills, RAG systems, frameworks
🏗️ **Transformers**: DeepSeek, Llama 4, Gemma 3, architecture trends

Try questions like:
- "How does step size affect optimization curves?"
- "What RAG skills do I need for AI agents?"
- "Compare DeepSeek and Llama 4 architectures"

What interests you most?"""

def create_architecture_comparison_viz():
    """Create transformer architecture comparison visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Model comparison data
    models = ['DeepSeek\nV3', 'Llama 4', 'Gemma 3', 'Qwen3\nMoE', 'Kimi K2']
    total_params = [671, 400, 27, 235, 1000]  # in billions
    active_params = [37, 17, 27, 22, 50]  # in billions (estimated for Kimi)
    
    # Plot 1: Total vs Active Parameters
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, total_params, width, label='Total Parameters', alpha=0.8, color='#FF6B6B')
    ax1.bar(x + width/2, active_params, width, label='Active Parameters', alpha=0.8, color='#4ECDC4')
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Parameters (Billions)', fontsize=12)
    ax1.set_title('2025 LLM Parameter Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Architecture Features
    features = ['MoE', 'MLA', 'GQA', 'Sliding\nWindow', 'Shared\nExpert']
    model_features = {
        'DeepSeek V3': [1, 1, 0, 0, 1],
        'Llama 4': [1, 0, 1, 0, 0],
        'Gemma 3': [0, 0, 1, 1, 0],
        'Qwen3 MoE': [1, 0, 1, 0, 0],
        'Kimi K2': [1, 1, 0, 0, 1]
    }
    
    # Create heatmap
    data = np.array(list(model_features.values()))
    im = ax2.imshow(data.T, cmap='YlOrRd', aspect='auto')
    
    ax2.set_xticks(np.arange(len(model_features)))
    ax2.set_yticks(np.arange(len(features)))
    ax2.set_xticklabels(list(model_features.keys()), rotation=45, ha='right')
    ax2.set_yticklabels(features)
    ax2.set_title('Architecture Features Comparison', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(model_features)):
        for j in range(len(features)):
            text = ax2.text(i, j, '✓' if data[i, j] else '✗',
                           ha="center", va="center", color="black", fontsize=12)
    
    plt.tight_layout()
    return fig

def create_streamlit_app():
    """Create comprehensive Streamlit app"""
    st.set_page_config(
        page_title="Ninth Ward God - AI Knowledge Hub", 
        layout="wide",
        page_icon="🧠",
        menu_items={
            'About': "Comprehensive AI resource covering Convex Optimization, AI Agents, and Transformer Architectures"
        }
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .topic-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #FF6B6B;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧠 Ninth Ward God - AI Knowledge Hub")
    st.markdown("*Your comprehensive resource for Convex Optimization, AI Agents, and Transformer Architectures*")
    st.markdown("---")
    
    # Initialize components
    analyzer = ConvexOptimizationCurves()
    chatbot = ComprehensiveChatbot()
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI Assistant", "📊 Convex Optimization", "🤖 AI Agents Guide", "🏗️ Transformer Architectures", "📚 Resources"])
    
    with tab1:
        st.header("Ask About Any Topic")
        
        # Topic selector buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎯 Convex Optimization", use_container_width=True):
                st.session_state.quick_question = "What are convex optimization curves?"
        with col2:
            if st.button("🤖 AI Agent Skills", use_container_width=True):
                st.session_state.quick_question = "What Python skills do I need for AI agents?"
        with col3:
            if st.button("🏗️ Transformers 2025", use_container_width=True):
                st.session_state.quick_question = "What are the trends in transformer architectures?"
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            welcome = chatbot._about_site()
            st.session_state.messages.append({"role": "assistant", "content": welcome})
        
        # Handle quick questions
        if 'quick_question' in st.session_state:
            question = st.session_state.quick_question
            del st.session_state.quick_question
            
            st.session_state.messages.append({"role": "user", "content": question})
            response = chatbot.answer_question(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about optimization, AI agents, or transformers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            response = chatbot.answer_question(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    
    with tab2:
        st.header("📊 Convex Optimization Curves")
        st.markdown("Based on 'Are Convex Optimization Curves Convex?' by Barzilai & Shamir (2025)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Interactive Visualization")
            
            eta_quad = st.slider("Step size η", 0.01, 1.0, 0.1, 0.01)
            x0_quad = st.slider("Initial point x₀", -5.0, 5.0, 3.0, 0.1)
            
            result = analyzer.quadratic_example(eta_quad, x0_quad)
            fig = analyzer.plot_optimization_curve(result, "f(x) = x²")
            st.pyplot(fig)
            
            L = 2.0
            st.info(f"**L = {L}, η/L = {eta_quad/L:.3f}**")
            if eta_quad <= 0.5:
                st.success("✅ η ≤ 1/L: Curve is guaranteed convex")
            else:
                st.warning("⚠️ η > 1/L: Curve may not be convex")
        
        with col2:
            st.subheader("Key Theorems")
            
            st.markdown("""
            <div class="topic-card">
            <h4>Theorem 1</h4>
            <p>For L-smooth convex functions with η ∈ (0, 1/L], the optimization curve is convex.</p>
            </div>
            
            <div class="topic-card">
            <h4>Theorem 2</h4>
            <p>For η ∈ (1.75/L, 2/L), the curve may not be convex despite convergence.</p>
            </div>
            
            <div class="topic-card">
            <h4>Theorem 3</h4>
            <p>Gradient norms always decrease monotonically for η ∈ (0, 2/L].</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.header("🤖 AI Agent Development Guide")
        
        st.markdown("""
        ### 🎯 Critical Skills for Production AI Agents
        
        Building AI Agents requires mastering several key areas. Start with hands-on projects and learn each component as you go.
        """)
        
        # Skills sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="topic-card">
            <h4>⚡ Python Skills</h4>
            <ul>
                <li><b>FastAPI</b> - Build lightweight APIs <a href="https://lnkd.in/dpADd3gh">📺</a></li>
                <li><b>Async Programming</b> - Handle concurrent tasks <a href="https://lnkd.in/dQDtam6S">📺</a></li>
                <li><b>Pydantic</b> - Data validation <a href="https://lnkd.in/dArUdWKT">📺</a></li>
                <li><b>Logging</b> - Debug complex agents <a href="https://lnkd.in/dYP3tWAk">📺</a></li>
                <li><b>Testing</b> - Unit & integration tests <a href="https://lnkd.in/d-sYNxxz">📺</a></li>
                <li><b>SQLAlchemy</b> - Database management <a href="https://lnkd.in/dj9UsCjd">📺</a></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="topic-card">
            <h4>🛠️ Frameworks & Monitoring</h4>
            <ul>
                <li><b>LangGraph</b> - Production agent framework <a href="https://lnkd.in/due46xmV">📺</a></li>
                <li><b>LangFuse</b> - Agent monitoring <a href="https://lnkd.in/dQbpvSVv">📺</a></li>
                <li><b>Prompt Engineering</b> - Optimize prompts <a href="https://lnkd.in/dAMk64iC">📖</a></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="topic-card">
            <h4>📚 RAG Skills</h4>
            <ul>
                <li><b>Understanding RAG</b> - Core concepts <a href="https://lnkd.in/dGUijEMw">📺</a></li>
                <li><b>Text Embeddings</b> - Search foundation <a href="https://lnkd.in/dtG7m-mv">📺</a></li>
                <li><b>Vector Database</b> - Efficient retrieval <a href="https://lnkd.in/dqbrK3d7">📺</a></li>
                <li><b>Chunking Strategies</b> - Smart splitting <a href="https://lnkd.in/dVRPRVfN">📺</a></li>
                <li><b>RAG with PostgreSQL</b> - Cost-effective <a href="https://lnkd.in/dDm7miwh">📺</a></li>
                <li><b>RAG with LangChain</b> - Quick prototyping <a href="https://lnkd.in/d3PuPKbF">📺</a></li>
                <li><b>RAG Evaluations</b> - Measure quality <a href="https://lnkd.in/dn-NDF_U">📺</a></li>
                <li><b>Advanced RAG</b> - Production tuning <a href="https://lnkd.in/dcXwX5Pp">📺</a></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Start Simple**: Build a basic FastAPI + LangChain chatbot
        2. **Add RAG**: Implement document retrieval with embeddings
        3. **Scale Up**: Add async processing and proper error handling
        4. **Monitor**: Implement logging and LangFuse monitoring
        5. **Optimize**: Fine-tune prompts and retrieval strategies
        """)
    
    with tab4:
        st.header("🏗️ Modern Transformer Architectures (2025)")
        st.markdown("Deep dive into state-of-the-art LLM architectures by Sebastian Raschka")
        
        # Architecture comparison visualization
        st.subheader("📊 Architecture Comparison")
        fig = create_architecture_comparison_viz()
        st.pyplot(fig)
        
        # Key models
        st.subheader("🔍 Key Models Analysis")
        
        model_tabs = st.tabs(["DeepSeek V3", "Llama 4", "Gemma 3", "Qwen3", "Kimi K2"])
        
        with model_tabs[0]:
            st.markdown("""
            ### DeepSeek V3/R1
            - **Scale**: 671B parameters (37B active)
            - **Key Innovation**: Multi-Head Latent Attention (MLA)
            - **MoE Design**: 256 experts, 9 active (1 shared + 8 selected)
            
            **Why It Matters**:
            - MLA compresses KV cache better than GQA
            - Shared expert learns common patterns
            - Foundation for reasoning model R1
            """)
        
        with model_tabs[1]:
            st.markdown("""
            ### Llama 4 Maverick
            - **Scale**: 400B parameters (17B active)
            - **Architecture**: Classic MoE with GQA
            - **MoE Design**: 2 active experts, alternating layers
            
            **Key Differences**:
            - Fewer but larger experts vs DeepSeek
            - Uses GQA instead of MLA
            - Alternates MoE and dense layers
            """)
        
        with model_tabs[2]:
            st.markdown("""
            ### Gemma 3
            - **Scale**: 27B parameters (all active)
            - **Key Innovation**: Sliding window attention
            - **Efficiency**: 40% KV cache reduction
            
            **Architecture Details**:
            - 5:1 local:global attention ratio
            - 1024 token sliding window
            - Both Pre and Post normalization
            """)
        
        with model_tabs[3]:
            st.markdown("""
            ### Qwen3 Family
            - **Sizes**: 0.6B to 235B (dense and MoE)
            - **Dense Models**: Deeper architectures
            - **MoE Models**: No shared expert
            
            **Notable Features**:
            - QK-Norm for training stability
            - Excellent performance across all sizes
            - 0.6B model great for local deployment
            """)
        
        with model_tabs[4]:
            st.markdown("""
            ### Kimi K2
            - **Scale**: 1 trillion parameters
            - **Architecture**: Enhanced DeepSeek V3
            - **Innovation**: First production Muon optimizer
            
            **Improvements over DeepSeek**:
            - 380 experts vs 256
            - Exceptional training stability
            - Currently largest open-weight model
            """)
        
        # Trends summary
        st.subheader("📈 2025 Architecture Trends")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="topic-card">
            <h4>MoE Adoption</h4>
            <p>Widespread use of Mixture-of-Experts for efficient scaling</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="topic-card">
            <h4>Attention Efficiency</h4>
            <p>MLA vs GQA vs Sliding Window approaches</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="topic-card">
            <h4>Core Stability</h4>
            <p>Transformer design remains stable with incremental improvements</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        st.header("📚 Resources & References")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📖 Papers & Articles
            - [Are Convex Optimization Curves Convex?](https://arxiv.org/abs/2503.10138)
            - [Sebastian Raschka's Transformer Analysis](https://lnkd.in/gk_z9Y_u)
            - [DeepSeek V3 Paper](https://arxiv.org/abs/2401.06066)
            
            ### 🎓 Learning Paths
            1. **Convex Optimization**: Start with interactive visualizations
            2. **AI Agents**: Follow the Python → RAG → Framework path
            3. **Transformers**: Understand MoE and attention mechanisms
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Tools & Frameworks
            - **LangChain**: RAG and agent development
            - **FastAPI**: Production API development
            - **LangGraph**: Agent orchestration
            - **LangFuse**: Monitoring and observability
            
            ### 💡 Quick Start Projects
            1. Build a simple RAG chatbot with LangChain
            2. Implement convex optimization visualizations
            3. Compare transformer architectures programmatically
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🧠 **Ninth Ward God** - Comprehensive AI Knowledge Hub | "
        "Topics: Convex Optimization, AI Agents, Transformer Architectures"
    )

if __name__ == "__main__":
    create_streamlit_app()