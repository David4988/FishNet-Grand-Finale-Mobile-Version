// Import WebLLM library
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const MODEL_URL_PREFIX = "./models/Llama-3.2-1B-instruct-q4f16_1-MLC/";

let engine = null;
let isInitialized = false;
let isLoading = false;

const messagesDiv = document.getElementById("messages");
const inputArea = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const statusText = document.getElementById("statusText");
const statusIndicator = document.getElementById("statusIndicator");

async function initializeEngine() {
    if (isInitialized) return;
    
    try {
        statusText.textContent = "Initializing AI model...";
        statusIndicator.className = "status-indicator loading";

        // Initialize the WebLLM engine with local model
        console.log("Creating WebLLM engine...");
        
        engine = new webllm.MLCEngine({
            appConfig: {
                model_list: [
                    {
                        model: MODEL_ID,
                        model_id: MODEL_ID,
                        model_lib: MODEL_URL_PREFIX + "Llama-3.2-1B-instruct-q4f16_1-MLC-webllm.wasm",
                        tokenizer_files: [
                            MODEL_URL_PREFIX + "tokenizer.json"
                        ],
                        model_params: [
                            MODEL_URL_PREFIX + "params_shard_0.bin",
                            MODEL_URL_PREFIX + "params_shard_1.bin",
                            MODEL_URL_PREFIX + "params_shard_2.bin",
                            MODEL_URL_PREFIX + "params_shard_3.bin",
                            MODEL_URL_PREFIX + "params_shard_4.bin",
                            MODEL_URL_PREFIX + "params_shard_5.bin",
                            MODEL_URL_PREFIX + "params_shard_6.bin",
                            MODEL_URL_PREFIX + "params_shard_7.bin",
                            MODEL_URL_PREFIX + "params_shard_8.bin",
                            MODEL_URL_PREFIX + "params_shard_9.bin",
                            MODEL_URL_PREFIX + "params_shard_10.bin",
                            MODEL_URL_PREFIX + "params_shard_11.bin",
                            MODEL_URL_PREFIX + "params_shard_12.bin",
                            MODEL_URL_PREFIX + "params_shard_13.bin",
                            MODEL_URL_PREFIX + "params_shard_14.bin",
                            MODEL_URL_PREFIX + "params_shard_15.bin",
                            MODEL_URL_PREFIX + "params_shard_16.bin",
                            MODEL_URL_PREFIX + "params_shard_17.bin",
                            MODEL_URL_PREFIX + "params_shard_18.bin",
                            MODEL_URL_PREFIX + "params_shard_19.bin",
                            MODEL_URL_PREFIX + "params_shard_20.bin",
                            MODEL_URL_PREFIX + "params_shard_21.bin",
                        ]
                    }
                ]
            }
        });

        // Listen for engine events
        engine.on("initStart", () => {
            statusText.textContent = "Loading model weights...";
            console.log("Model initialization started");
        });

        engine.on("initProgress", (report) => {
            const percent = Math.round((report.progress / report.total) * 100);
            statusText.textContent = `Loading: ${percent}%`;
            console.log(`Loading progress: ${percent}%`);
        });

        engine.on("initEnd", () => {
            statusText.textContent = "Ready to chat ✓";
            statusIndicator.className = "status-indicator";
            isInitialized = true;
            inputArea.disabled = false;
            sendBtn.disabled = false;
            
            // Clear init message and show greeting
            messagesDiv.innerHTML = "";
            addMessage("Hi! I'm Llama 3.2 1B running entirely offline. How can I help you today?", "assistant");
            console.log("Model initialized successfully");
        });

        // Initialize the engine (this loads the model)
        console.log("Loading model:", MODEL_ID);
        await engine.init(MODEL_ID);

    } catch (error) {
        console.error("Engine initialization error:", error);
        statusText.textContent = "Error loading model";
        statusIndicator.className = "status-indicator";
        
        showError(`Failed to initialize model: ${error.message}`);
        console.error("Full error details:", error);
    }
}

function addMessage(content, role) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTyping() {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message assistant";
    messageDiv.id = "typing-message";
    
    messageDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTyping() {
    const typingMsg = document.getElementById("typing-message");
    if (typingMsg) {
        typingMsg.remove();
    }
}

function showError(message) {
    const errorDiv = document.createElement("div");
    errorDiv.className = "error-message";
    errorDiv.textContent = message;
    messagesDiv.appendChild(errorDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
    if (!isInitialized || isLoading) return;
    
    const message = inputArea.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(message, "user");
    inputArea.value = "";
    inputArea.style.height = "auto";
    
    isLoading = true;
    sendBtn.disabled = true;
    inputArea.disabled = true;
    showTyping();
    
    try {
        // Call LLM
        const response = await engine.chat.completions.create({
            messages: [
                {
                    role: "user",
                    content: message
                }
            ],
            max_tokens: 512,
            temperature: 0.7,
        });
        
        removeTyping();
        
        const assistantMessage = response.choices[0].message.content;
        addMessage(assistantMessage, "assistant");
        
    } catch (error) {
        console.error("Chat error:", error);
        removeTyping();
        showError(`Error generating response: ${error.message}`);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        inputArea.disabled = false;
        inputArea.focus();
    }
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);

inputArea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

inputArea.addEventListener("input", () => {
    inputArea.style.height = "auto";
    inputArea.style.height = Math.min(inputArea.scrollHeight, 100) + "px";
});

// Initialize on load
window.addEventListener("load", () => {
    initializeEngine();
});

// Fallback timeout in case initialization takes too long
setTimeout(() => {
    if (!isInitialized) {
        statusText.textContent = "Model initialization taking longer than expected...";
    }
}, 60000);
