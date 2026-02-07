import React from 'react';
import { CTASection, CTAContainer, CTATitle, CTASubtitle, CTAButton } from './CTA.styles';

const CTA: React.FC = () => {
  return (
    <CTASection>
      <CTAContainer
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
      >
        <CTATitle>Secure your financial future.</CTATitle>
        <CTASubtitle>
          Join 50+ financial institutions using Sentinel AI to stop fraud in its tracks. 
          Start your risk-free assessment today.
        </CTASubtitle>
        <CTAButton
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.98 }}
        >
          Get Started Now
        </CTAButton>
      </CTAContainer>
    </CTASection>
  );
};

export default CTA;