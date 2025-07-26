import React, { useState } from 'react';
import styles from './TextInput.module.css';

function TextInput({ onSend, isLoading }) {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSend(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.inputContainer}>
      <div className={styles.inputWrapper}>
        <input
          type="text"
          placeholder="Type your shopping request... (e.g., 'I need red Nike shoes under $100')"
          className={styles.textInput}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        <button 
          onClick={handleSend} 
          className={`${styles.sendButton} ${isLoading ? styles.loading : ''}`}
          disabled={!input.trim() || isLoading}
        >
          {isLoading ? (
            <div className={styles.spinner}></div>
          ) : (
            '📤 Send'
          )}
        </button>
      </div>
      
      <div className={styles.suggestions}>
        <span className={styles.suggestionLabel}>Try saying:</span>
        <div className={styles.suggestionTags}>
          <button 
            className={styles.suggestionTag}
            onClick={() => setInput("Show me Nike sneakers under $150")}
            disabled={isLoading}
          >
            Nike sneakers under $150
          </button>
          <button 
            className={styles.suggestionTag}
            onClick={() => setInput("I need a red dress for party")}
            disabled={isLoading}
          >
            Red dress for party
          </button>
          <button 
            className={styles.suggestionTag}
            onClick={() => setInput("Show me laptops with good battery")}
            disabled={isLoading}
          >
            Laptops with good battery
          </button>
        </div>
      </div>
    </div>
  );
}

export default TextInput;