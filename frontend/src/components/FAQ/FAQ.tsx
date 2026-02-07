import React, { useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Section, AccordionContainer, AccordionItem, Question, Answer } from './FAQ.styles';
import { Title, HeaderArea } from '../Features/Features.styles';

const FAQ: React.FC = () => {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const faqs = [
    {
      q: "How secure is my transaction data?",
      a: "All uploaded files are encrypted using AES-256 at rest and TLS 1.3 in transit. We prioritize data privacy and never store unencrypted sensitive information."
    },
    {
      q: "Which AI models are used for detection?",
      a: "Sentinel uses a sophisticated ensemble of XGBoost, Random Forest, and Logistic Regression models to ensure high accuracy while maintaining interpretability."
    },
    {
      q: "Is the system AML/KYC compliant?",
      a: "Yes. Our rule engine is built specifically to align with global AML/KYC regulations, providing automated flags for suspicious activity."
    }
  ];

  return (
    <Section id="faq">
      <HeaderArea>
        <Title>Common <span>Questions.</span></Title>
      </HeaderArea>

      <AccordionContainer>
        {faqs.map((faq, i) => (
          <AccordionItem key={i}>
            <Question onClick={() => setActiveIndex(activeIndex === i ? null : i)}>
              <h4>{faq.q}</h4>
              <span>{activeIndex === i ? '−' : '+'}</span>
            </Question>
            <AnimatePresence>
              {activeIndex === i && (
                <Answer
                  initial={{ height: 0, opacity: 0, marginTop: 0 }}
                  animate={{ height: 'auto', opacity: 1, marginTop: 16 }}
                  exit={{ height: 0, opacity: 0, marginTop: 0 }}
                >
                  {faq.a}
                </Answer>
              )}
            </AnimatePresence>
          </AccordionItem>
        ))}
      </AccordionContainer>
    </Section>
  );
};

export default FAQ;