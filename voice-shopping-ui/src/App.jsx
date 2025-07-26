import React, { useState } from 'react';
import styles from './App.module.css';
import VoiceInput from './Components/VoiceInput';
import ChatBox from './Components/ChatBox';
import Recommendation from './Components/Recommendation';
import TextInput from './Components/TextInput';
import { sendTextToAssistant, resetSession } from './api/assistant';

function App() {
  const [messages, setMessages] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [searchLinks, setSearchLinks] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  const handleUserText = async (text) => {
    if (!text.trim()) return;
    
    setIsLoading(true);
    try {
      // Add user message immediately
      setMessages(prev => [...prev, { user: text, assistant: null }]);
      
      const response = await sendTextToAssistant(text);
      
      // Update the last message with assistant response
      setMessages(prev => 
        prev.map((msg, index) => 
          index === prev.length - 1 ? { ...msg, assistant: response.reply } : msg
        )
      );
      
      setRecommendations(response.recommendations || []);
      setSearchLinks(response.search_urls || {});
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => 
        prev.map((msg, index) => 
          index === prev.length - 1 
            ? { ...msg, assistant: "Sorry, I encountered an error. Please try again." } 
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await resetSession();
      setMessages([]);
      setRecommendations([]);
      setSearchLinks({});
    } catch (error) {
      console.error('Error resetting session:', error);
    }
  };

  return (
    <div className={styles.appContainer}>
      <header className={styles.header}>
        <h1 className={styles.title}>🛒 Shopper Assist</h1>
        <p className={styles.subtitle}>Your AI-powered shopping companion</p>
      </header>

      <div className={styles.mainContent}>
        <div className={styles.inputSection}>
          <VoiceInput onTranscribe={handleUserText} />
          <TextInput onSend={handleUserText} isLoading={isLoading} />
        </div>

        <ChatBox messages={messages} isLoading={isLoading} />
        
        {(recommendations.length > 0 || Object.keys(searchLinks).length > 0) && (
          <Recommendation 
            recommendations={recommendations} 
            searchLinks={searchLinks} 
          />
        )}
        
        {messages.length > 0 && (
          <div className={styles.resetSection}>
            <button onClick={handleReset} className={styles.resetButton}>
              🔄 Reset Chat
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;