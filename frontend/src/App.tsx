import React from 'react';
import styled from 'styled-components';
import Header from './components/Header/Header';
import Hero from './components/Hero/Hero';
import Features from './components/Features/Features';
import Process from './components/Process/Process';
import ModelPerformance from './components/ModelPerformance/ModelPerformance';
import FAQ from './components/FAQ/FAQ';
import CTA from './components/CTA/CTA';
import Footer from './components/Footer/Footer';

const MainWrapper = styled.main`
  position: relative;
  width: 100%;
`;

function App() {
  return (
    <MainWrapper>
      <Header />
      <Hero />
      <Features />
      <Process />
      <ModelPerformance />
      <FAQ />
      <CTA />
      <Footer />
    </MainWrapper>
  );
}

export default App;