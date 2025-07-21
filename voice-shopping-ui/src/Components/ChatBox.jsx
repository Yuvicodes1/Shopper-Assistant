import React from 'react';
import styles from './ChatBox.module.css';

function ChatBox({ messages }) {
  return (
    <div className={styles.chatContainer}>
      {messages.length === 0 ? (
        <div className={styles.placeholder}>No messages yet...</div>
      ) : (
        messages.map((m, i) => (
          <div key={i} className={styles.messageBlock}>
            <span className={styles.user}>You:</span>
            <span className={styles.text}>{m.user}</span>
            <span className={styles.assistant}>Assistant:</span>
            <span className={styles.text}>{m.assistant}</span>
          </div>
        ))
      )}
    </div>
  );
}

export default ChatBox;
