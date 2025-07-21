import React from "react";
import styles from "./Recommendation.module.css";

const Recommendation = ({ recommendations }) => {
  return (
    <div className={styles.recommendationContainer}>
      <h2 className={styles.title}>💡 Smart Recommendations</h2>
      <ul className={styles.list}>
        {recommendations.map((rec, index) => (
          <li key={index} className={styles.item}>{rec}</li>
        ))}
      </ul>

      <h2 className={styles.title}>🔗 Quick Shop Links</h2>
      <ul className={styles.links}>
        <li><a href="https://www.ajio.com" target="_blank" rel="noreferrer">ajio</a></li>
        <li><a href="https://www.amazon.in" target="_blank" rel="noreferrer">amazon</a></li>
        <li><a href="https://www.flipkart.com" target="_blank" rel="noreferrer">flipkart</a></li>
        <li><a href="https://www.myntra.com" target="_blank" rel="noreferrer">myntra</a></li>
      </ul>
    </div>
  );
};

export default Recommendation;
