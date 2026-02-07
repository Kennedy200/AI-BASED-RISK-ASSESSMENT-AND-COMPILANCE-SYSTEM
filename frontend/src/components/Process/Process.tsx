import React from 'react';
import { motion } from 'framer-motion';
import { FiZap, FiFileText, FiDatabase } from 'react-icons/fi';
import { 
  ProcessSection, 
  Container, 
  TimelineLine, 
  ActiveLine,
  StepRow, 
  StepNumber, 
  StepContent, 
  StepTitle, 
  StepDescription,
  StepImage,
  FileIcon,
  ModelChip,
  ShapBar
} from './Process.styles';
import { Title, HeaderArea } from '../Features/Features.styles';

// --- Mini Visual Components ---

const IngestionVisual = () => (
  <div style={{ 
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'center', 
    gap: '40px', 
    width: '100%', 
    height: '100%',
    position: 'relative' 
  }}>
    {/* CSV File - Centered Left */}
    <FileIcon 
      $color="#2563eb"
      style={{ position: 'relative' }} 
      initial={{ x: -30, opacity: 0 }}
      whileInView={{ x: 0, opacity: 1, rotate: -10 }}
      animate={{ y: [0, -10, 0] }}
      transition={{ 
        x: { duration: 0.5 },
        opacity: { duration: 0.5 },
        y: { duration: 2, repeat: Infinity, ease: "easeInOut" }
      }}
    >
      <FiFileText size={24} color="#2563eb" />
      <span style={{ fontSize: '10px', fontWeight: 800 }}>.CSV</span>
    </FileIcon>

    {/* Central Processing Node */}
    <motion.div 
      initial={{ scale: 0 }}
      whileInView={{ scale: 1 }}
      style={{ 
        width: '44px', 
        height: '44px', 
        borderRadius: '50%', 
        background: 'rgba(37, 99, 235, 0.1)',
        border: '1px solid #e2e8f0',
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        zIndex: 2
      }}
    >
      <FiZap color="#2563eb" />
    </motion.div>

    {/* XLSX File - Centered Right */}
    <FileIcon 
      $color="#10b981"
      style={{ position: 'relative' }} 
      initial={{ x: 30, opacity: 0 }}
      whileInView={{ x: 0, opacity: 1, rotate: 10 }}
      animate={{ y: [0, 10, 0] }}
      transition={{ 
        x: { duration: 0.5, delay: 0.2 },
        opacity: { duration: 0.5, delay: 0.2 },
        y: { duration: 2, delay: 0.5, repeat: Infinity, ease: "easeInOut" }
      }}
    >
      <FiDatabase size={24} color="#10b981" />
      <span style={{ fontSize: '10px', fontWeight: 800 }}>.XLSX</span>
    </FileIcon>
  </div>
);

const AnalysisVisual = () => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
    <ModelChip animate={{ scale: [1, 1.05, 1] }} transition={{ duration: 2, repeat: Infinity }}>
      <span /> XGBOOST CLASSIFIER
    </ModelChip>
    <ModelChip animate={{ scale: [1, 1.05, 1] }} transition={{ duration: 2, delay: 0.3, repeat: Infinity }}>
      <span /> RANDOM FOREST
    </ModelChip>
    <ModelChip animate={{ scale: [1, 1.05, 1] }} transition={{ duration: 2, delay: 0.6, repeat: Infinity }}>
      <span /> LOGISTIC REGRESSION
    </ModelChip>
  </div>
);

const IntelligenceVisual = () => (
  <div style={{ background: 'white', padding: '20px', borderRadius: '12px', boxShadow: 'var(--shadow-sm)', width: '240px' }}>
    <p style={{ fontSize: '10px', fontWeight: 700, marginBottom: '10px', color: '#64748b' }}>SHAP FEATURE IMPORTANCE</p>
    <ShapBar $width="80%" $positive={true} initial={{ width: 0 }} whileInView={{ width: '80%' }} />
    <ShapBar $width="60%" $positive={true} initial={{ width: 0 }} whileInView={{ width: '60%' }} />
    <ShapBar $width="40%" $positive={false} initial={{ width: 0 }} whileInView={{ width: '40%' }} />
    <div style={{ borderTop: '1px solid #f1f5f9', marginTop: '10px', paddingTop: '10px', display: 'flex', justifyContent: 'space-between' }}>
       <span style={{ fontSize: '10px', fontWeight: 800, color: '#ef4444' }}>RISK: 92%</span>
       <span style={{ fontSize: '10px', fontWeight: 800, color: '#2563eb' }}>DETAILS</span>
    </div>
  </div>
);

const Process: React.FC = () => {
  const steps = [
    {
      title: "Smart Data Ingestion",
      description: "Upload your transaction logs in Excel or CSV. Our AI automatically maps your columns using fuzzy matching, reducing manual setup time by 90%.",
      visual: <IngestionVisual />
    },
    {
      title: "Ensemble AI Analysis",
      description: "Your data is processed through our triple-model ensemble. Every transaction is scored against historical patterns and real-time compliance rules.",
      visual: <AnalysisVisual />
    },
    {
      title: "Actionable Intelligence",
      description: "Review flagged transactions with clear SHAP explanations. Resolve alerts, export regulatory reports, and manage your risk perimeter from one dashboard.",
      visual: <IntelligenceVisual />
    }
  ];

  return (
    <ProcessSection id="process">
      <HeaderArea>
        <Title>The Sentinel <span>Workflow.</span></Title>
        <p style={{ color: '#64748b', fontSize: '1.1rem' }}>
          Three simple steps to transition from raw data to regulatory-grade fraud detection.
        </p>
      </HeaderArea>

      <Container>
        <TimelineLine>
          <ActiveLine 
            initial={{ scaleY: 0 }}
            whileInView={{ scaleY: 1 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
          />
        </TimelineLine>

        {steps.map((step, index) => (
          <StepRow 
            key={index}
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: index * 0.1 }}
          >
            <StepNumber className="active">
              {index + 1}
            </StepNumber>
            <StepContent>
              <StepTitle>{step.title}</StepTitle>
              <StepDescription>{step.description}</StepDescription>
              <StepImage>
                {step.visual}
              </StepImage>
            </StepContent>
          </StepRow>
        ))}
      </Container>
    </ProcessSection>
  );
};

export default Process;