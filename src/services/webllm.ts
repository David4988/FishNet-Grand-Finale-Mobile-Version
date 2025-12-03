// Simple Offline Chat Service - No downloads, instant responses
import { databaseService, FishCatch } from "./database";

export interface WebLLMInitParams {
  modelPath?: string;
  preferredModel?: string;
  appName?: string;
  onProgress?: (progress: any) => void;
}

class WebLLMChatService {
  private isInitialized = false;

  /**
   * Initialize - instant, no downloads
   */
  async initialize(params: WebLLMInitParams = {}): Promise<void> {
    console.log("[OfflineChat] Ready to chat - no downloads needed!");
    this.isInitialized = true;
  }

  /**
   * Generate chat response using template-based system
   * Fast, offline, no ML model needed
   */
  async generate(
    userMessage: string,
    systemPrompt: string = "",
    language: string = "en"
  ): Promise<string> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const response = await this.generateResponse(userMessage, systemPrompt, language);
      return response;
    } catch (error) {
      console.error("[OfflineChat] Error:", error);
      throw error;
    }
  }

  /**
   * Generate intelligent response based on keywords and context
   */
  private async generateResponse(
    userMessage: string,
    systemPrompt: string,
    language: string
  ): Promise<string> {
    const lowerMessage = userMessage.toLowerCase().trim();

    // Get user's catch history for context
    let context = "";
    try {
      const catches = await databaseService.getAllCatches();
      if (catches.length > 0) {
        context = `User has ${catches.length} recorded catches. `;
      }
    } catch (e) {
      // Ignore errors
    }

    // Response templates based on keywords
    const responses: Record<string, Record<string, string>> = {
      en: {
        greeting: "Hello! I'm your offline fishing assistant. How can I help you today? Ask me about fish species, fishing techniques, or your catches!",
        fish: "Fish species vary by region and season. Common Indian fish include: Rohu, Catla, Pomfret, Carp, and Tilapia. They thrive in different water conditions. Which region are you fishing in?",
        technique: "Effective fishing techniques depend on the species and water type. Popular methods include: Rod fishing, Net fishing, Trap fishing, and Hook fishing. Each has pros and cons based on your target species.",
        weather: "Weather greatly affects fishing! Best times: Early morning or late evening when water is cooler. Avoid midday heat. Overcast days are excellent. Rain increases fish activity in fresh water.",
        location: "Good fishing locations in India: Rivers (Ganges, Brahmaputra), Lakes, Reservoirs, Coastal areas, and Backwaters. Local knowledge is crucial - ask fishermen in your area!",
        help: "I can help you with: Fish identification, Fishing techniques, Best times to fish, Location suggestions, Equipment advice, and Weather tips. What interests you?",
        default: "That's an interesting question! Based on fishing knowledge: Most successful fishermen combine patience, technique, and understanding fish behavior. Would you like specific advice about techniques or species?",
      },
      hi: {
        greeting: "नमस्ते! मैं आपका ऑफ़लाइन मछली पकड़ने का सहायक हूँ। आप मुझसे मछली की प्रजातियों, तकनीकों या अपनी पकड़ के बारे में पूछ सकते हैं।",
        fish: "भारत में आम मछलियों में रोहू, कतला, पोम्फ्रेट, कार्प और तिलपिया शामिल हैं। ये विभिन्न जल स्थितियों में पनपती हैं। आप किस क्षेत्र में मछली पकड़ रहे हैं?",
        technique: "प्रभावी तकनीकें: रॉड मछली पकड़ना, जाल मछली पकड़ना, जाल मछली पकड़ना, और हुक मछली पकड़ना। प्रत्येक की अपनी विशेषताएं हैं।",
        weather: "सुबह जल्दी या शाम को मछली पकड़ना सबसे अच्छा है। बादल वाले दिन बेहतरीन हैं। बारिश से मछली सक्रिय होती है।",
        location: "भारत में अच्छी मछली पकड़ने की जगहें: गंगा, ब्रह्मपुत्र नदियां, झीलें, जलाशय, तटीय क्षेत्र। स्थानीय ज्ञान महत्वपूर्ण है।",
        help: "मैं आपकी सहायता कर सकता हूँ: मछली की पहचान, तकनीकें, सही समय, स्थान सुझाव आदि। आप क्या जानना चाहते हैं?",
        default: "यह एक अच्छा सवाल है! सफल मछुआरे धैर्य, तकनीक और मछली के व्यवहार को समझने को मिलाते हैं।",
      },
      ta: {
        greeting: "வணக்கம்! நான் உங்கள் ஆஃப்லைன் மீன்பிடி உதவியாளர். மீன் வகைகள், பிடிக்கும் நுட்பங்கள் பற்றி கேளுங்கள்.",
        fish: "இந்தியায் உள்ள பொதுவான மீன்: ரோஹு, கட்ல, பம்ப்ரெட், கார்ப், திலாபியா.",
        technique: "பயனுள தொழில்நுட்பங்கள்: கம்பு மீன்பிடி, வலை மீன்பிடி, கொக்கி மீன்பிடி.",
        weather: "பொதுவாக காலை அல்லது மாலை சிறந்தது. மேகமூட்ட நாட்கள் சிறப்பு.",
        location: "நல்ல மீன்பிடி இடங்கள்: ஆறுகள், குளங்கள், நீர்க்கட்டளைகள்.",
        help: "நான் உதவ முடியும்: மீன் அடையாளம், நுட்பங்கள், சரியான நேரம்.",
        default: "நல்ல கேள்வி! வெற்றிகரமான மீன்பिடிக்காரர்கள் பொறுமை மற்றும் தொழில்நுட்பத்தை கூட்டுகிறார்கள்.",
      },
      te: {
        greeting: "హలో! నేను మీ ఆఫ్‌లైన్ చేపల పట్టే సహాయకుడిని. చేపల జాతులు, పద్ధతుల గురించి అడగండి.",
        fish: "భారతదేశంలో సాధారణ చేపలు: రోహు, కట్ల, పెంప్ఫ్రెట్, కార్ప్, టిలాపియా.",
        technique: "ప్రభావవంతమైన పద్ధతులు: రాడ్ చేపల పట్టుకోవడం, జాల చేపల పట్టుకోవడం, కొక్కుచేపల పట్టుకోవడం.",
        weather: "సాధారణంగా ఉదయం లేదా సాయంత్రం ఉత్తమం. చెరువు కూడా గొప్ప.",
        location: "మంచి చేపల పట్టే ప్రదేశాలు: నదులు, సరస్సులు, జలాశయాలు.",
        help: "నేను సహాయం చేయగలను: చేపల గుర్తింపు, పద్ధతులు, సరిఅయిన సమయం.",
        default: "మంచి ప్రశ్న! విజయవంతమైన చేపల పట్టేవారు ఓపిక మరియు పద్ధతిని కలుపుతారు.",
      },
    };

    // Get language-specific responses
    const lang = language || "en";
    const langResponses = responses[lang] || responses.en;

    // Match keywords and return appropriate response
    if (lowerMessage.includes("hello") || lowerMessage.includes("hi") || lowerMessage.includes("start")) {
      return langResponses.greeting;
    } else if (lowerMessage.includes("fish") || lowerMessage.includes("species") || lowerMessage.includes("type")) {
      return langResponses.fish;
    } else if (lowerMessage.includes("technique") || lowerMessage.includes("method") || lowerMessage.includes("how")) {
      return langResponses.technique;
    } else if (lowerMessage.includes("weather") || lowerMessage.includes("time") || lowerMessage.includes("when")) {
      return langResponses.weather;
    } else if (lowerMessage.includes("where") || lowerMessage.includes("location") || lowerMessage.includes("place")) {
      return langResponses.location || langResponses.default;
    } else if (lowerMessage.includes("help") || lowerMessage.includes("what can")) {
      return langResponses.help;
    }

    // Default response
    return langResponses.default;
  }
}

export const webllmChatService = new WebLLMChatService();

/**
 * Format and process messages with instructions
 * Format: "instruction,$prompt"
 * Example: "keep the message short,$what is rohu fish?"
 * Returns: { instruction, prompt, language }
 */
export function parseMessageFormat(message: string): { instruction: string; prompt: string; language: string } {
  const parts = message.split(',$');
  
  if (parts.length === 2) {
    // Format matched: "instruction,$prompt"
    const instruction = parts[0].trim();
    const prompt = parts[1].trim();
    
    return {
      instruction,
      prompt,
      language: 'en' // default language
    };
  }
  
  // No special format, treat entire message as prompt
  return {
    instruction: 'default',
    prompt: message,
    language: 'en'
  };
}

/**
 * Process and translate response
 * Returns: { translation, results }
 */
export async function formatResponseWithTranslation(
  response: string,
  language: string = 'en',
  instruction: string = 'default'
): Promise<{ translation: string; results: string; language: string }> {
  // Apply instruction-based formatting
  let formattedResponse = response;
  
  if (instruction.toLowerCase().includes('short')) {
    // Short but meaningful: take up to 2 sentences, ensure last ends cleanly
    const sentences = response
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(Boolean);
    if (sentences.length === 0) {
      formattedResponse = response.trim();
    } else {
      const take = sentences.slice(0, Math.min(2, sentences.length));
      formattedResponse = take.join(' ');
      // Ensure it ends with punctuation
      if (!/[.!?]$/.test(formattedResponse)) formattedResponse += '.';
    }
  } else if (instruction.toLowerCase().includes('detailed')) {
    // Detailed instruction - keep full response
    formattedResponse = response;
  } else if (instruction.toLowerCase().includes('bullet')) {
    // Bullet points format
    const points = response.split(/(?<=[.!?])\s+/).filter(s => s.trim());
    formattedResponse = points.map(s => `• ${s.trim()}`).join('\n');
  } else {
    // Default: keep short (2 sentences max)
    const sentences = response
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(Boolean);
    const take = sentences.slice(0, Math.min(2, sentences.length));
    formattedResponse = take.join(' ');
    if (!/[.!?]$/.test(formattedResponse) && formattedResponse.length) formattedResponse += '.';
  }
  
  return {
    translation: language !== 'en' ? `[${language.toUpperCase()}]` : '[EN]',
    results: formattedResponse,
    language
  };
}

/**
 * Build system prompt with user context
 */
export async function buildFishingContextPrompt(language: string = "en"): Promise<string> {
  const prompts: Record<string, string> = {
    en: "You are a helpful fishing assistant. Answer questions about fishing techniques, fish species, weather, and locations. Be concise and practical.",
    hi: "आप एक मददगार मछली पकड़ने के सहायक हैं। मछली पकड़ने की तकनीकों, मछली की प्रजातियों, मौसम और स्थानों के बारे में प्रश्नों का उत्तर दें।",
    ta: "நீங்கள் ஒரு உதவிகரமான மீன் பிடி உதவியாளர். மீன் பிடி நுட்பங்கள், மீன் வகைகள், வானிலை மற்றும் இடங்கள் பற்றி கேள்விகளுக்கு பதிலளிக்கவும்.",
    te: "మీరు ఒక సహాయక చేపల పట్టే సహాయకుడి. చేపల పట్టే పద్ధతులు, చేపల జాతులు, వానిలై మరియు ప్రదేశాల గురించి ప్రశ్నలకు సమాధానం ఇవ్వండి.",
  };

  return prompts[language] || prompts.en;
}
