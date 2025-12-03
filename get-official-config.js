/**
 * Fetch actual model configuration from WebLLM defaults
 */

import fetch from 'node-fetch';

async function getModelInfo() {
  try {
    // WebLLM appConfig is served from the official repo
    const response = await fetch(
      "https://raw.githubusercontent.com/mlc-ai/web-llm/main/src/config.json"
    );
    
    if (!response.ok) {
      console.error("Failed to fetch config:", response.status);
      return;
    }
    
    const config = await response.json();
    
    console.log('\n═══════════════════════════════════════════════════════════');
    console.log('WebLLM Official Configuration');
    console.log('═══════════════════════════════════════════════════════════\n');
    
    // Find Phi-3-mini model
    const phiModels = config.model_list.filter(m => 
      m.model_id.includes('Phi-3-mini') || m.model_id.includes('phi-3-mini')
    );
    
    console.log('✅ Phi-3-mini Models Found:\n');
    phiModels.forEach(model => {
      console.log(`Model ID: ${model.model_id}`);
      console.log(`Model URL: ${model.model}`);
      console.log(`Lib URL: ${model.model_lib}`);
      console.log();
    });
    
    // Also show TinyLlama as fallback
    const tinyModels = config.model_list.filter(m => 
      m.model_id.includes('TinyLlama')
    );
    
    if (tinyModels.length > 0) {
      console.log('✅ TinyLlama Model (Fallback):\n');
      tinyModels.slice(0, 1).forEach(model => {
        console.log(`Model ID: ${model.model_id}`);
        console.log(`Model URL: ${model.model}`);
        console.log(`Lib URL: ${model.model_lib}`);
        console.log();
      });
    }
    
  } catch (error) {
    console.error("Error:", error.message);
  }
}

getModelInfo();
