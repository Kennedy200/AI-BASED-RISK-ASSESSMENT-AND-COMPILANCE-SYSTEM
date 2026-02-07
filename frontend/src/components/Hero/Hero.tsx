import React from 'react';
import ParticleBackground from './ParticleBackground';
import { 
  HeroSection, 
  Container, 
  ContentWrapper, 
  Badge, 
  Headline, 
  Subheadline, 
  ButtonGroup, 
  HeroButton,
  ImageWrapper,
  HeroImage,
  BackgroundBlob,
  FloatingCard,
  CardHeader,
  CardTitle,
  ProgressBar,
  CardMeta
} from './Hero.styles';

const Hero: React.FC = () => {
  return (
    <HeroSection>
      <ParticleBackground />
      
      {/* Background Blobs */}
      <BackgroundBlob 
        $color="rgba(37, 99, 235, 0.1)" 
        $top="-10%" 
        $left="-10%" 
        animate={{ y: [0, 40, 0], x: [0, 20, 0] }}
        transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
      />
      <BackgroundBlob 
        $color="rgba(139, 92, 246, 0.1)" 
        $top="40%" 
        $left="80%" 
        style={{ width: '600px', height: '600px' }}
        animate={{ y: [0, -60, 0], x: [0, -30, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />

      <Container>
        <ContentWrapper
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Badge>AI v2.0 Live</Badge>
          <Headline>
            Fraud detection <br />
            <span>without the friction.</span>
          </Headline>
          <Subheadline>
            Stop financial crime in real-time with our enterprise-grade AI risk assessment platform. 
            Reduce false positives by 40% starting today.
          </Subheadline>
          <ButtonGroup>
            <HeroButton whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              Get Started
            </HeroButton>
          </ButtonGroup>
        </ContentWrapper>

        {/* The Real Image + Glass Card */}
        <ImageWrapper
           initial={{ opacity: 0, x: 50 }}
           animate={{ opacity: 1, x: 0 }}
           transition={{ duration: 0.8, delay: 0.2 }}
        >
          {/* Main Hero Image from Unsplash (Data/Tech Concept) */}
          <HeroImage 
            src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2670&auto=format&fit=crop" 
            alt="Fraud Detection Dashboard" 
            initial={{ scale: 0.98 }}
            animate={{ scale: 1 }}
            transition={{ duration: 3, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" }}
          />

          {/* Floating 'Live Alert' Card */}
          <FloatingCard
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.8, duration: 0.5 }}
            style={{ y: -10 }} // Subtle floating visual
          >
            <CardHeader>
              <span>Suspicious Activity</span>
              <span>High Risk</span>
            </CardHeader>
            <CardTitle>Transaction #8821</CardTitle>
            <ProgressBar>
              <div />
            </ProgressBar>
            <CardMeta>
              <span>Score: 92/100</span>
              <span>Loc: Lagos, NG</span>
            </CardMeta>
          </FloatingCard>
        </ImageWrapper>
      </Container>
    </HeroSection>
  );
};

export default Hero;