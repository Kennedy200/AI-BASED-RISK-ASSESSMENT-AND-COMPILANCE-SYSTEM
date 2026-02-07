import React from 'react';
import { FiCpu, FiShield, FiBarChart2 } from 'react-icons/fi';
import { 
  Section, 
  HeaderArea, 
  Title, 
  Subtitle, 
  Grid, 
  FeatureCard, 
  IconWrapper, 
  FeatureTitle, 
  FeatureDescription 
} from './Features.styles';

const Features: React.FC = () => {
  const featureList = [
    {
      icon: <FiCpu size={24} />,
      title: "Real-time AI Detection",
      description: "Our ensemble of 3 ML models (XGBoost, RF, LR) analyzes transactions in under 50ms with 99.9% accuracy."
    },
    {
      icon: <FiShield size={24} />,
      title: "Regulatory Compliance",
      description: "Automated AML and KYC rule engine ensures you stay compliant with global banking regulations out of the box."
    },
    {
      icon: <FiBarChart2 size={24} />,
      title: "Explainable Insights",
      description: "Every risk score comes with SHAP-based explanations, showing you exactly why a transaction was flagged."
    }
  ];

  return (
    <Section id="features">
      <HeaderArea>
        <Title>Enterprise-grade <span>protection.</span></Title>
        <Subtitle>
          We’ve built a comprehensive suite of tools designed to help financial institutions 
          identify, monitor, and mitigate risk without slowing down growth.
        </Subtitle>
      </HeaderArea>

      <Grid>
        {featureList.map((f, index) => (
          <FeatureCard 
            key={index}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.2 }}
          >
            <IconWrapper>{f.icon}</IconWrapper>
            <FeatureTitle>{f.title}</FeatureTitle>
            <FeatureDescription>{f.description}</FeatureDescription>
          </FeatureCard>
        ))}
      </Grid>
    </Section>
  );
};

export default Features;