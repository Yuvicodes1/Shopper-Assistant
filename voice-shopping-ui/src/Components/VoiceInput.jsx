import React from "react";
import styles from "./VoiceInput.module.css";

const VoiceInput = ({ startListening, stopListening, resetTranscript }) => {
  return (
    <div className={styles.voiceInputContainer}>
      <button onClick={startListening} className={styles.button}>🎙 Start Voice</button>
      <button onClick={stopListening} className={styles.button}>🛑 Stop & Send</button>
      <button onClick={resetTranscript} className={styles.resetButton}>🔁 Reset</button>
    </div>
  );
};

export default VoiceInput;