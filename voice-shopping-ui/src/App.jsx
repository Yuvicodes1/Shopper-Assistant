import React, { useState } from 'react';
import styles from './App.module.css'; // ✅ Use module CSS, not .css
import VoiceInput from './Components/VoiceInput';
import ChatBox from './Components/ChatBox';
import Recommendation from './Components/Recommendation';
import TextInput from './Components/TextInput';
import { sendTextToAssistant, resetSession } from './api/assistant';

function App() {
  const [messages, setMessages] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [searchLinks, setSearchLinks] = useState({});

  const handleUserText = async (text) => {
    if (!text.trim()) return;
    const response = await sendTextToAssistant(text);
    setMessages(prev => [...prev, { user: text, assistant: response.reply }]);
    setRecommendations(response.recommendations);
    setSearchLinks(response.search_urls);
  };

  const handleReset = async () => {
    await resetSession();
    setMessages([]);
    setRecommendations([]);
    setSearchLinks({});
  };

  return (
    <div className={styles.appContainer}>
      <h1 className={styles.title}>🛒 Shopper Assist</h1>

      <div className={styles.controls}>
        <VoiceInput onTranscribe={handleUserText} onReset={handleReset} />
      </div>

      <TextInput onSend={handleUserText} />
      <ChatBox messages={messages} />
      <Recommendation recommendations={recommendations} searchLinks={searchLinks} />
    </div>
  );
}

export default App;
