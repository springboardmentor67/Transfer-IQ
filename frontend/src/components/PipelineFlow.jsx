import React from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  MessageSquareHeart,
  ShieldPlus,
  Brain,
  Cpu,
  Target,
  ArrowRight,
} from 'lucide-react';

const STEPS = [
  { key: 'performance', label: 'Performance Data', icon: Activity, color: '#38bdf8' },
  { key: 'sentiment', label: 'Sentiment Data', icon: MessageSquareHeart, color: '#34d399' },
  { key: 'injury', label: 'Injury Data', icon: ShieldPlus, color: '#fbbf24' },
  { key: 'lstm', label: 'LSTM Model', icon: Brain, color: '#a78bfa' },
  { key: 'xgboost', label: 'XGBoost Model', icon: Cpu, color: '#6366f1' },
  { key: 'prediction', label: 'Final Prediction', icon: Target, color: '#fb7185' },
];

function PipelineFlow({ activeStep = 5 }) {
  return (
    <div className="glass pipeline-wrap">
      <div className="section-head-row">
        <h3 className="section-title">AI Pipeline</h3>
        <span className="section-kicker">Data to Model to Prediction to Insights</span>
      </div>

      <div className="pipeline-grid">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isActive = idx === activeStep;

          return (
            <React.Fragment key={step.key}>
              <motion.div
                className={`pipeline-card ${isActive ? 'active' : ''}`}
                whileHover={{ y: -4, scale: 1.01 }}
                transition={{ duration: 0.2 }}
                style={{ '--step-color': step.color }}
              >
                <div className="pipeline-icon-box">
                  <Icon size={18} />
                </div>
                <div className="pipeline-label">{step.label}</div>
              </motion.div>

              {idx < STEPS.length - 1 && (
                <div className="pipeline-arrow" aria-hidden="true">
                  <ArrowRight size={16} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

export default PipelineFlow;
