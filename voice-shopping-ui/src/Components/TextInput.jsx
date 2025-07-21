import React, { useState } from 'react';
import styles from './TextInput.module.css';

function TextInput({ onSend }) {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim()) {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div className={styles.inputContainer}>
      <input
        type="text"
        placeholder="Type your request..."
        className={styles.textInput}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
      />
      <button onClick={handleSend} className={styles.sendButton}>Send</button>
    </div>
  );
}

export default TextInput;
