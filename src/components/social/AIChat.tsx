import { useState, useEffect, useRef } from 'react';
import { preferencesService } from '@/services/preferences';
import { Send, Bot, Loader2, AlertCircle, Download } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { authService } from '@/services/auth';
import { webllmChatService, buildFishingContextPrompt, parseMessageFormat, formatResponseWithTranslation } from '@/services/webllm';
import { cn } from '@/lib/utils';

interface AIMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface DownloadProgress {
  loaded: number;
  total: number;
  percent: number;
}

export const AIChat = () => {
    // Map language codes to full names
    const languageNames: Record<string, string> = {
      en: 'English',
      hi: 'Hindi',
      ta: 'Tamil',
      te: 'Telugu',
      kn: 'Kannada',
      ml: 'Malayalam',
      gu: 'Gujarati',
      mwr: 'Marwari',
      bn: 'Bengali',
      pa: 'Punjabi',
      mr: 'Marathi',
      or: 'Odia',
    };
  const { t, i18n } = useTranslation();
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState(authService.getState().user);
  const [engine, setEngine] = useState<any>(null);
  const [modelReady, setModelReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [progressText, setProgressText] = useState('Initializing AI model...');
  const [settingsLanguage, setSettingsLanguage] = useState('en');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unsubscribeAuth = authService.subscribe((state) => {
      setCurrentUser(state.user);
      if (state.user) {
        loadChatHistory(state.user.email);
      }
    });
    const unsubscribePrefs = preferencesService.subscribe((prefs) => {
      setSettingsLanguage(prefs.language || 'en');
    });
    // Initial load
    if (currentUser?.email) {
      loadChatHistory(currentUser.email);
    }
    setSettingsLanguage(preferencesService.getPreferences().language || 'en');
    return () => {
      unsubscribeAuth();
      unsubscribePrefs();
    };
  }, []);

  // Update the fixed welcome message when language changes
  useEffect(() => {
    // Only update if the first message is the fixed welcome
    if (messages.length > 0 && messages[0].id === 'welcome' && messages[0].role === 'assistant') {
      setMessages((prev) => {
        const updated = [...prev];
        updated[0] = { ...updated[0], content: getWelcomeMessage(), timestamp: new Date() };
        saveChatHistory(currentUser?.email, updated);
        return updated;
      });
      // Ensure scroll reflects updated content
      setTimeout(() => scrollToBottom(), 0);
    }
  }, [settingsLanguage]);

  // Initialize WebLLM engine and model on mount
  useEffect(() => {
    let isMounted = true;
    async function initializeEngine() {
      setIsInitializing(true);
      setProgressText('Loading WebLLM...');
      try {
        const lib = await import('https://esm.run/@mlc-ai/web-llm');
        setProgressText('Creating engine...');
        const modelName = 'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC';
        const engineInstance = await lib.CreateMLCEngine(modelName, {
          initProgressCallback: (report: any) => {
            if (report.text) setProgressText(report.text);
            setDownloadProgress({
              loaded: report.progress,
              total: report.total,
              percent: report.total ? (report.progress / report.total) * 100 : 0
            });
          }
        });
        if (!isMounted) return;
        setEngine(engineInstance);
        setModelReady(true);
        setIsInitializing(false);
        setProgressText('Ready!');
        if (messages.length === 0) {
          setMessages([{
            id: 'welcome',
            role: 'assistant',
            content: getWelcomeMessage(),
            timestamp: new Date(),
          }]);
        }
      } catch (e: any) {
        setInitError(e.message || 'Failed to initialize AI engine');
        setModelReady(false);
        setIsInitializing(false);
        setProgressText('Error');
      }
    }
    initializeEngine();
    return () => { isMounted = false; };
  }, [currentUser]);
  // Local chat history per Gmail
  const getHistoryKey = (email?: string) => email ? `ai_chat_history_${email}` : 'ai_chat_history_default';

  const loadChatHistory = (email?: string) => {
    try {
      const key = getHistoryKey(email);
      const stored = localStorage.getItem(key);
      if (stored) {
        const parsed = JSON.parse(stored);
        setMessages(parsed.map((msg: any) => ({ ...msg, timestamp: new Date(msg.timestamp) })));
      } else {
        setMessages([{
          id: 'welcome',
          role: 'assistant',
          content: getWelcomeMessage(),
          timestamp: new Date(),
        }]);
      }
    } catch (err) {
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: getWelcomeMessage(),
        timestamp: new Date(),
      }]);
    }
  };

  const saveChatHistory = (email?: string, msgs?: AIMessage[]) => {
    try {
      const key = getHistoryKey(email);
      localStorage.setItem(key, JSON.stringify(msgs || messages));
    } catch (err) {
      // Ignore
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const getWelcomeMessage = () => {
    const welcomeMessages: Record<string, string> = {
      en: "🎣 Welcome to Offline AI Fishing Assistant! I'm powered by a local language model running entirely in your browser. I can help you identify fish species, improve your fishing techniques, analyze your recent catches, and answer any fishing questions. Type your question to get started!",
      hi: "🎣 ऑफ़लाइन AI मछली पकड़ने सहायक में आपका स्वागत है! मैं आपके ब्राउज़र में पूरी तरह चल रहे एक स्थानीय भाषा मॉडल द्वारा संचालित हूं। मैं मछली की प्रजातियों की पहचान करने, आपकी मछली पकड़ने की तकनीकों में सुधार करने, आपकी हाल की पकड़ का विश्लेषण करने और किसी भी मछली पकड़ने के सवाल का जवाब देने में मदद कर सकता हूं।",
      ta: "🎣 ஆஃப்லைன் AI மீன்பிடி உதவியாளரில் வரவேற்கிறோம்! நான் உங்கள் ப்ரௌசரில் முழுமையாக இயங்கும் உள்ளூர் மொழி மாதிரியில் இயக்கப்படுகிறேன். மீன்களின் இனங்களை அடையாளம் காண, உங்கள் மீன்பிடி நுட்பங்களை மேம்படுத்த, சமீபத்திய பிடிப்புகளை பகுப்பாய்வு செய்ய, அல்லது எந்தவொரு மீன்பிடி கேள்விக்கு பதிலளிக்க உதவுகிறேன்.",
      ml: "🎣 ഓഫ്‌ലൈനിലെ AI ഫിഷിംഗ് അസിസ്റ്റന്റിലേക്ക് സ്വാഗതം! നിങ്ങളുടെ ബ്രൗസറിൽ പൂർണ്ണമായും പ്രവർത്തിക്കുന്ന ഒരു ലോക്കൽ ഭാഷ മോഡലാണ് എന്നെ പ്രവർത്തിപ്പിക്കുന്നത്. മത്സ്യ ഇനങ്ങളെ തിരിച്ചറിയാൻ, നിങ്ങളുടെ മീൻപിടിത്ത വിദ്യകൾ മെച്ചപ്പെടുത്താൻ, പുതിയ പിടികൾ വിശകലനം ചെയ്യാൻ, മീൻപിടിത്തവുമായി ബന്ധപ്പെട്ട ഏതൊരു ചോദ്യത്തിനും ഉത്തരമൊരുക്കാൻ ഞാൻ സഹായിക്കും.",
      te: "🎣 ఆఫ్లైన్ AI ఫిషింగ్ అసిస్టెంట్‌కు స్వాగతం! మీ బ్రౌజర్‌లో పూర్తిగా నడిచే లోకల్ లాంగ్వేజ్ మోడల్‌తో నేను పనిచేస్తున్నాను. చేపల జాతులను గుర్తించడంలో, మీ చేపల వేట పద్ధతులను మెరుగుపరచడంలో, మీ తాజా పట్టులను విశ్లేషించడంలో, మరియు చేపల వేటపై ఉన్న ప్రశ్నలకు సమాధానాలు ఇవ్వడంలో నేను సహాయం చేయగలను.",
      kn: "🎣 ಆಫ್‌ಲೈನ್ AI ಮೀನುಗಾರಿಕೆ ಸಹಾಯಕಕ್ಕೆ ಸ್ವಾಗತ! ನಿಮ್ಮ ಬ್ರೌಸರ್‌ನಲ್ಲಿ ಪೂರ್ಣವಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸುವ ಸ್ಥಳೀಯ ಭಾಷಾ ಮಾದರಿಯಿಂದ ನಾನು ಚಾಲಿತನಾಗಿದ್ದೇನೆ. ಮೀನು ಜಾತಿಗಳನ್ನು ಗುರುತಿಸಲು, ನಿಮ್ಮ ಮೀನುಗಾರಿಕೆ ತಂತ್ರಗಳನ್ನು ಉತ್ತಮಗೊಳಿಸಲು, ನಿಮ್ಮ ಇತ್ತೀಚಿನ ಹಿಡಿತಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಲು ಮತ್ತು ಮೀನುಗಾರಿಕೆಯ ಬಗ್ಗೆ ಯಾವುದೇ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಲು ನಾನು ಸಹಾಯ ಮಾಡಬಹುದು.",
      gu: "🎣 ઓફલાઇન AI ફિશિંગ અસિસ્ટન્ટમાં આપનું સ્વાગત છે! હું તમારા બ્રાઉઝરમાં સંપૂર્ણપણે ચાલતા સ્થાનિક ભાષા મોડેલ દ્વારા સંચાલિત છું. હું તમને માછલીની જાતિઓ ઓળખવામાં, તમારી માછલી પકડવાની તકનીકો સુધારવામાં, તમારી તાજેતરની પકડનું વિશ્લેષણ કરવામાં અને માછલી પકડવા સંબંધિત કોઈપણ પ્રશ્નોના જવાબ આપવા માટે મદદ કરી શકું છું.",
      mr: "🎣 ऑफलाइन AI फिशिंग असिस्टंट मध्ये आपले स्वागत आहे! मी तुमच्या ब्राउजरमध्ये पूर्णपणे चालणाऱ्या स्थानिक भाषा मॉडेलवर चालतो. मी माशांच्या जाती ओळखण्यात, तुमच्या मासेमारी तंत्र सुधारण्यात, तुमच्या अलीकडील पकडीचे विश्लेषण करण्यात आणि मासेमारीबाबत कोणत्याही प्रश्नांची उत्तरे देण्यात मदत करू शकतो.",
      pa: "🎣 ਆਫਲਾਈਨ AI ਫਿਸ਼ਿੰਗ ਅਸਿਸਟੈਂਟ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ! ਮੈਂ ਤੁਹਾਡੇ ਬ੍ਰਾਉਜ਼ਰ ਵਿੱਚ ਪੂਰੀ ਤਰ੍ਹਾਂ ਚੱਲਦੇ ਇੱਕ ਲੋਕਲ ਭਾਸ਼ਾ ਮਾਡਲ ਨਾਲ ਚੱਲਦਾ ਹਾਂ। ਮੈਂ ਮੱਛੀਆਂ ਦੀਆਂ ਕਿਸਮਾਂ ਦੀ ਪਹਿਚਾਣ ਕਰਨ, ਤੁਹਾਡੀਆਂ ਮੱਛੀ ਫੜਨ ਦੀਆਂ ਤਕਨੀਕਾਂ ਨੂੰ ਸੁਧਾਰਨ, ਤੁਹਾਡੇ ਹਾਲੀਆ ਫੜ ਨੂੰ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਨ ਅਤੇ ਮੱਛੀ ਫੜਨ ਬਾਰੇ ਕਿਸੇ ਵੀ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ।",
      or: "🎣 ଅଫଲାଇନ AI ମାଛ ଧରା ସହାୟକକୁ ସ୍ବାଗତ! ମୁଁ ଆପଣଙ୍କର ବ୍ରାଉଜରରେ ସମ୍ପୂର୍ଣ୍ଣ ଭାବେ ଚାଲୁ ଥିବା ଏକ ସ୍ଥାନୀୟ ଭାଷା ମଡେଲ୍ ଦ୍ୱାରା ସଞ୍ଚାଳିତ ହୁଏ। ମୁଁ ମାଛର ଜାତି ପରିଚୟ କରିବା, ଆପଣଙ୍କର ମାଛ ଧରା ପ୍ରକିୟାକୁ ଉନ୍ନତ କରିବା, ଆପଣଙ୍କର ସମ୍ପ୍ରତିଧରା ମାଛଗୁଡ଼ିକୁ ବିଶ୍ଲେଷଣ କରିବା, ଏବଂ ମାଛ ଧରା ସମ୍ବନ୍ଧିତ କ any ଣସି ପ୍ରଶ୍ନର ଉତ୍ତର ଦେବାରେ ସାହାଯ୍ୟ କରିପାରେ.",
      bn: "🎣 অফলাইন AI ফিশিং অ্যাসিস্ট্যান্টে স্বাগতম! আমি আপনার ব্রাউজারে সম্পূর্ণভাবে চলা একটি লোকাল ভাষা মডেলের দ্বারা চালিত। আমি মাছের প্রজাতি শনাক্ত করতে, আপনার মাছ ধরার কৌশল উন্নত করতে, আপনার সাম্প্রতিক ধরা বিশ্লেষণ করতে এবং মাছ ধরার সম্পর্কিত যেকোনো প্রশ্নের উত্তর দিতে সাহায্য করতে পারি।",
      mwr: "🎣 ऑफलाइन AI फिशिंग असिस्टेंट में आप रो स्वागत है! थारो ब्राउजर में पूरू चलण वालो स्थानीय भाषा मॉडल सूं मैं चालू हूं. मैं माछलां की जात पहचानवा, थारी बणावट सुधारवा, हाल की पकड़ की पड़ताल करवा, अर माछी पकड़ण बारे में कोई भी सवाल को जवाब देवा में मदद करूं सूं."
    };
    return welcomeMessages[settingsLanguage] || welcomeMessages.en;
  };

  const handleSendMessage = async () => {
    if (!newMessage.trim() || isLoading || !engine) return;
    if (!modelReady || isInitializing) return;

    // Parse message format (but don't use instruction if not explicitly set)
    const hasFormat = newMessage.includes(',$');
    let userDisplayContent = newMessage;
    let instruction = 'default';

    if (hasFormat) {
      const parsed = parseMessageFormat(newMessage);
      userDisplayContent = parsed.prompt;
      instruction = parsed.instruction;
    }

    const userMsg: AIMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: userDisplayContent,
      timestamp: new Date(),
    };
    setMessages((prev) => {
      const updated = [...prev, userMsg];
      saveChatHistory(currentUser?.email, updated);
      return updated;
    });
    setNewMessage('');
    setIsLoading(true);
    try {
      // Get current language from settings (most reliable)
      const currentLang = settingsLanguage || 'en';
      
      console.log('[AIChat] Using language:', currentLang);
      
      // Always use English prompt for the initial generation to ensure fastest and most accurate response
      const basePrompt = 'You are a fishing assistant. Answer in English.';
      // Requesting a single concise paragraph ensures it fits naturally within the 100 token limit without cutting off
      const systemPrompt = `${basePrompt} Respond in one concise paragraph (max 3-4 sentences) that fully answers the question.`;

      // Direct model call (no template fallback) with concise constraints
      // Optimized for speed but allowing enough length for completeness
      const res = await engine.chat.completions.create({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userDisplayContent }
        ],
        max_tokens: 100, // Reduced to 100 as requested
        temperature: 0.7,
        repetition_penalty: 1.2, // Keeps loops away
        top_p: 0.9
      });
      let responseContent: string = res.choices?.[0]?.message?.content || 'Unable to generate response.';
      
      console.log('[AIChat] Initial English response:', responseContent);

      // Language Translation Step
      // If language is not English, translate the response using MyMemory API (better quality than local LLM)
      if (currentLang !== 'en') {
        try {
          // Map language codes to MyMemory format (usually 2-letter code)
          // MyMemory supports: hi, ta, te, kn, ml, gu, bn, pa, mr
          const targetLangCode = currentLang; 
          const encodedText = encodeURIComponent(responseContent);
          const url = `https://api.mymemory.translated.net/get?q=${encodedText}&langpair=en|${targetLangCode}`;
          
          const apiRes = await fetch(url);
          const data = await apiRes.json();
          
          if (data && data.responseData && data.responseData.translatedText) {
             responseContent = data.responseData.translatedText;
             console.log('[AIChat] Translated response (MyMemory):', responseContent);
          } else {
             console.warn('[AIChat] MyMemory translation returned no text:', data);
          }
        } catch (transError) {
          console.error('Translation failed, falling back to English:', transError);
        }
      }

      // Ensure the response ends with punctuation to look complete
      let finalContent = responseContent.trim();
      if (finalContent && !/[.!?]$/.test(finalContent)) {
         finalContent += '.';
      }

      const aiMsg: AIMessage = {
        id: `ai_${Date.now()}`,
        role: 'assistant',
        content: finalContent,
        timestamp: new Date(),
      };
      setMessages((prev) => {
        const updated = [...prev, aiMsg];
        saveChatHistory(currentUser?.email, updated);
        return updated;
      });
    } catch (error: any) {
      const errorMessages: Record<string, string> = {
        en: `Sorry, error: ${error.message || error}`,
        hi: 'क्षमा करें, प्रतिक्रिया उत्पन्न करते समय मुझे त्रुटि आई। कृपया दोबारा प्रयास करें।',
        ta: 'மன்னிக்கவும், பதிலை உत்பత்தி செய்ய பிழை ஏற்பட்டது। மீண்டும் முயற்சி செய்யவும்।',
        te: 'క్షమించండి, ప్రతిస్పందన ఉత్పత్తిలో లోపం ఏర్పడింది. మళ్లీ ప్రయత్నించండి.',
        kn: 'ಕ್ಷಮಿಸಿ, ಪ್ರತಿಕ್ರಿಯೆ ಉತ್ಪಾದಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ಪುನಃ ಪ್ರಯತ್ನಿಸಿ.',
        ml: 'ക്ഷമിക്കണം, പ്രതികരണം ഉത്പാദിപ്പിക്കുമ്പോൾ പിശക് സംഭവിച്ചു. വീണ്ടും ശ്രമിക്കുക.',
        gu: 'માફ કરશો, પ્રતિક્રિયા પેદા કરતી વખતે ભૂલ થઈ. ફરીથી પ્રયાસ કરો.',
        bn: 'দুঃখিত, প্রতিক্রিয়া তৈরিতে ত্রুটি ঘটেছে। আবার চেষ্টা করুন।',
        pa: 'ਮਾਫ਼ ਕਰਨਾ, ਜਵਾਬ ਤਿਆਰ ਕਰਨ ਵਿੱਚ ਗਲਤੀ ਸੀ। ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
        mr: 'क्षमा करा, प्रतिक्रिया तयार करताना त्रुटी झाली. पुन्हा प्रयत्न करा.'
      };

      const errorMsg: AIMessage = {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: errorMessages[settingsLanguage] || errorMessages.en,
        timestamp: new Date(),
      };
      setMessages((prev) => {
        const updated = [...prev, errorMsg];
        saveChatHistory(currentUser?.email, updated);
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (isInitializing) {
    return (
      <div className="flex flex-col h-full bg-background items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
          {/* Simplified initializing UI without download text or progress bar */}
          <p className="text-muted-foreground text-sm">Preparing AI Assistant…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="border-b bg-card/50 backdrop-blur-sm p-4">
        <div className="flex items-center gap-3">
          <div className="bg-primary p-2 rounded-full">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-foreground">
              {settingsLanguage === 'en' ? 'Offline AI Fishing Assistant'
                : settingsLanguage === 'hi' ? 'ऑफलाइन AI मछली सहायक'
                : settingsLanguage === 'ta' ? 'ஆஃப்லைன் AI மீன்பிடி உதவியாளர்'
                : 'Offline AI Fishing Assistant'}
              <span className="ml-2 text-xs text-muted-foreground">({languageNames[settingsLanguage] || 'English'})</span>
            </h3>
            <p className="text-xs text-muted-foreground">
              {modelReady
                ? settingsLanguage === 'en'
                  ? 'Runs entirely offline • Responds in your language'
                  : settingsLanguage === 'hi'
                    ? 'पूरी तरह ऑफलाइन चलता है • आपकी भाषा में प्रतिक्रिया'
                  : settingsLanguage === 'ta'
                    ? 'முழுமையாக ஆஃப்லைன் • உங்கள் மொழியில் பதிலளிக்கிறது'
                  : 'Runs entirely offline • Responds in your language'
                : settingsLanguage === 'en'
                  ? 'Model initialization failed'
                  : settingsLanguage === 'hi'
                    ? 'मॉडल प्रारंभिकरण विफल'
                  : settingsLanguage === 'ta'
                    ? 'மாதிரி துவக்கம் தோல்வியடைந்தது'
                  : 'Model initialization failed'}
            </p>
          </div>
          <Download className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>

      {initError && (
        <Alert variant="destructive" className="m-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{initError}</AlertDescription>
        </Alert>
      )}

      <ScrollArea className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4 max-w-3xl mx-auto">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                'flex gap-3 items-start',
                message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              )}
            >
              <Avatar
                className={cn(
                  'h-8 w-8',
                  message.role === 'assistant' && 'bg-primary'
                )}
              >
                <AvatarFallback
                  className={cn(
                    message.role === 'assistant'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-primary/10 text-primary'
                  )}
                >
                  {message.role === 'assistant' ? (
                    <Bot className="h-4 w-4" />
                  ) : (
                    currentUser?.name.slice(0, 2).toUpperCase()
                  )}
                </AvatarFallback>
              </Avatar>

              <div
                className={cn(
                  'flex-1 px-4 py-3 rounded-2xl max-w-[80%]',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground ml-auto'
                    : 'bg-muted/50'
                )}
              >
                <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 items-start">
              <Avatar className="h-8 w-8 bg-primary">
                <AvatarFallback className="bg-primary text-primary-foreground">
                  <Bot className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 px-4 py-3 rounded-2xl bg-muted/50 max-w-[80%]">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <div className="border-t bg-background p-4">
        <div className="flex gap-2 max-w-3xl mx-auto">
          <Input
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              isInitializing
                ? settingsLanguage === 'en'
                  ? 'Loading model...'
                  : settingsLanguage === 'hi'
                    ? 'मॉडल लोड हो रहा है...'
                  : settingsLanguage === 'ta'
                    ? 'மாதிரி ஏற்றப்படுகிறது...'
                  : 'Loading model...'
                : settingsLanguage === 'en'
                  ? 'Ask me anything about fishing... (responds in your language)'
                  : settingsLanguage === 'hi'
                    ? 'मुझसे मछली पकड़ने के बारे में कोई भी सवाल पूछें...'
                  : settingsLanguage === 'ta'
                    ? 'மீன்பிடி பற்றி என்னிடம் ஏதும் கேளுங்கள்... (உங்கள் மொழியில் பதிலளிக்கிறது)'
                  : 'Ask me anything about fishing... (responds in your language)'
            }
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!newMessage.trim() || isLoading}
            className="bg-primary hover:bg-primary/90 text-primary-foreground"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        {!modelReady && !downloadProgress && (
          <p className="text-xs text-destructive mt-2">
            {settingsLanguage === 'en'
              ? '⚠️ Offline model is not ready. Chat functionality is unavailable.'
              : settingsLanguage === 'hi'
                ? '⚠️ ऑफलाइन मॉडल तैयार नहीं है। चैट कार्यक्षमता अनुपलब्ध है।'
              : settingsLanguage === 'ta'
                ? '⚠️ ஆஃப்லைன் மாதிரி தயாராக இல்லை. அரட்டை செயல்பாடு கிடைக்கவில்லை.'
              : '⚠️ Offline model is not ready. Chat functionality is unavailable.'}
          </p>
        )}
      </div>
    </div>
  );
};
