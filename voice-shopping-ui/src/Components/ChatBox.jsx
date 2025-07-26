import React, { useEffect, useRef } from 'react';
import styles from './ChatBox.module.css';

function ChatBox({ messages, isLoading }) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className={styles.chatContainer}>
      {messages.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>💬</div>
          <h3 className={styles.emptyTitle}>Ready to help you shop!</h3>
          <p className={styles.emptyText}>
            Start by typing or speaking your shopping request. I can help you find products, 
            compare prices, and discover the best deals across multiple stores.
          </p>
          <div className={styles.exampleQueries}>
            <div className={styles.exampleTitle}>Example queries:</div>
            <ul className={styles.exampleList}>
              <li>"Show me blue Nike running shoes under $120"</li>
              <li>"I need a formal shirt for office wear"</li>
              <li>"Find me wireless headphones with good battery life"</li>
            </ul>
          </div>
        </div>
      ) : (
        <div className={styles.messagesList}>
          {messages.map((message, index) => (
            <div key={index} className={styles.messageBlock}>
              <div className={styles.userMessage}>
                <div className={styles.messageHeader}>
                  <span className={styles.userLabel}>👤 You</span>
                  <span className={styles.timestamp}>
                    {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
                <div className={styles.messageContent}>
                  {message.user}
                </div>
              </div>
              
              <div className={styles.assistantMessage}>
                <div className={styles.messageHeader}>
                  <span className={styles.assistantLabel}>🤖 Assistant</span>
                  {message.assistant && (
                    <span className={styles.timestamp}>
                      {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  )}
                </div>
                <div className={styles.messageContent}>
                  {message.assistant ? (
                    message.assistant
                  ) : (
                    <div className={styles.typingIndicator}>
                      <div className={styles.typingDots}>
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                      <span className={styles.typingText}>Thinking...</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  );
}

export default ChatBox;