import React from 'react';
import { 
  Section, 
  Container, 
  MetricsColumn, 
  MetricCard, 
  MetricHeader, 
  ProgressBarBase, 
  ProgressBarFill,
  TableColumn,
  ComparisonTable
} from './ModelPerformance.styles';
import { Title, HeaderArea } from '../Features/Features.styles';

const ModelPerformance: React.FC = () => {
  const ensembleMetrics = [
    { label: "Ensemble Accuracy", value: "99.94%" },
    { label: "Precision Score", value: "92.10%" },
    { label: "Recall Score", value: "85.45%" },
    { label: "F1 Score", value: "88.65%" }
  ];

  const tableData = [
    { model: "XGBoost", acc: "99.95%", pre: "90%", rec: "85%", speed: "42ms" },
    { model: "Random Forest", acc: "99.91%", pre: "80%", rec: "75%", speed: "120ms" },
    { model: "Log. Regression", acc: "95.20%", pre: "10%", rec: "90%", speed: "12ms" },
  ];

  return (
    <Section id="performance">
      <HeaderArea>
        <Title>Proven <span>Performance.</span></Title>
        <p style={{ color: '#64748b', fontSize: '1.1rem' }}>
          Our models are trained on over 284,000 transactions with 30+ engineered features 
          to achieve industry-leading detection rates.
        </p>
      </HeaderArea>

      <Container>
        {/* Left: Ensemble Highlights */}
        <MetricsColumn>
          {ensembleMetrics.map((m, i) => (
            <MetricCard 
              key={i}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <MetricHeader>
                <h4>{m.label}</h4>
                <span>{m.value}</span>
              </MetricHeader>
              <ProgressBarBase>
                <ProgressBarFill 
                  $width={m.value}
                  initial={{ scaleX: 0, originX: 0 }}
                  whileInView={{ scaleX: 1 }}
                  transition={{ duration: 1, delay: 0.5 }}
                />
              </ProgressBarBase>
            </MetricCard>
          ))}
        </MetricsColumn>

        {/* Right: Detailed Comparison */}
        <TableColumn
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h3 style={{ marginBottom: '30px', fontWeight: 800 }}>Detailed Benchmark</h3>
          <ComparisonTable>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Recall</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, i) => (
                <tr key={i}>
                  <td>{row.model}</td>
                  <td>{row.acc}</td>
                  <td>{row.rec}</td>
                  <td>{row.speed}</td>
                </tr>
              ))}
            </tbody>
          </ComparisonTable>
        </TableColumn>
      </Container>
    </Section>
  );
};

export default ModelPerformance;