import streamlit as st
import asyncio
import threading
import queue
import websockets
import struct
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Config / Constants
# ---------------------------
DEPTH_WSS = "wss://depth-api-feed.dhan.co/twentydepth"
DEPTH_20_REQ = 23

# Alert thresholds - adjust based on your trading style
ALERT_CONFIG = {
    'large_order_threshold': 1000,  # Quantity threshold for large orders
    'imbalance_ratio_threshold': 3.0,  # Bid/Ask imbalance ratio
    'spread_compression_threshold': 0.1,  # % spread compression
    'iceberg_detection_threshold': 5,  # Number of replenishments to detect iceberg
    'volume_spike_threshold': 2.0,  # Multiple of average volume
    'price_level_break_threshold': 5,  # Number of levels broken through
    'order_flow_momentum_threshold': 0.7,  # Order flow momentum score
}

INSTITUTIONAL_DETECTION_CONFIG = {
    # Nifty-specific settings (adjust for other instruments)
    'nifty_lot_size': 75,
    'institutional_single_order_threshold': 300,  # 4+ lots in single order
    'institutional_quantity_threshold': 750,      # 10+ lots total
    'retail_fragmentation_threshold': 0.7,        # 70% of orders are lot-sized
    'institutional_order_concentration_threshold': 0.3,  # 30% of quantity in few orders
    'block_deal_threshold': 7500,                # 100+ lots (potential block deal)
    'order_size_variance_threshold': 2.0,        # High variance indicates mixed player types
}

# ---------------------------
# Alert Detection Classes
# ---------------------------

class AlertEngine:
    """Sophisticated alert detection engine for market depth analysis"""
    
    def __init__(self):
        self.price_history = {}  # security_id -> deque of price points
        self.volume_history = {}  # security_id -> deque of volume data
        self.order_flow_history = {}  # security_id -> deque of order flow data
        self.last_depth = {}  # security_id -> last depth snapshot
        self.alerts = deque(maxlen=100)  # Store recent alerts
        self.iceberg_tracker = {}  # Track potential iceberg orders
        
    def analyze_depth_and_generate_alerts(self, security_id: str, bid_data: List[Dict], ask_data: List[Dict]) -> List[Dict]:
        """Main analysis function - returns list of alerts for this update"""
        alerts = []
        timestamp = datetime.now()
        
        # Initialize tracking for new security
        if security_id not in self.price_history:
            self._initialize_security_tracking(security_id)
            
        # Calculate key metrics
        metrics = self._calculate_metrics(security_id, bid_data, ask_data)
        
        # Run alert detection algorithms
        alerts.extend(self._detect_large_orders(security_id, bid_data, ask_data, timestamp))
        alerts.extend(self._detect_order_imbalance(security_id, metrics, timestamp))
        alerts.extend(self._detect_spread_compression(security_id, metrics, timestamp))
        alerts.extend(self._detect_iceberg_orders(security_id, bid_data, ask_data, timestamp))
        alerts.extend(self._detect_volume_spikes(security_id, metrics, timestamp))
        alerts.extend(self._detect_level_breaks(security_id, bid_data, ask_data, timestamp))
        alerts.extend(self._detect_order_flow_momentum(security_id, metrics, timestamp))
        alerts.extend(self.detect_institutional_patterns(security_id, bid_data, ask_data, timestamp))
        
        # Update history
        self._update_history(security_id, metrics)
        self.last_depth[security_id] = {'bid': bid_data, 'ask': ask_data, 'timestamp': timestamp}
        
        # Add alerts to global queue
        for alert in alerts:
            self.alerts.append(alert)
            
        return alerts
    
    def _initialize_security_tracking(self, security_id: str):
        """Initialize tracking structures for a new security"""
        self.price_history[security_id] = deque(maxlen=100)
        self.volume_history[security_id] = deque(maxlen=100)
        self.order_flow_history[security_id] = deque(maxlen=50)
        self.iceberg_tracker[security_id] = {}
        
    def _calculate_metrics(self, security_id: str, bid_data: List[Dict], ask_data: List[Dict]) -> Dict:
        """Calculate key market microstructure metrics"""
        if not bid_data or not ask_data:
            return {}
            
        best_bid = max(bid_data, key=lambda x: x['price'])
        best_ask = min(ask_data, key=lambda x: x['price'])
        
        # Basic metrics
        spread = best_ask['price'] - best_bid['price']
        mid_price = (best_bid['price'] + best_ask['price']) / 2
        spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        # Volume metrics
        total_bid_qty = sum(level['quantity'] for level in bid_data)
        total_ask_qty = sum(level['quantity'] for level in ask_data)
        total_bid_orders = sum(level['orders'] for level in bid_data)
        total_ask_orders = sum(level['orders'] for level in ask_data)
        
        # Imbalance metrics
        qty_imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) if (total_bid_qty + total_ask_qty) > 0 else 0
        order_imbalance = (total_bid_orders - total_ask_orders) / (total_bid_orders + total_ask_orders) if (total_bid_orders + total_ask_orders) > 0 else 0
        
        # Depth metrics (top 5 levels)
        top5_bid_qty = sum(level['quantity'] for level in sorted(bid_data, key=lambda x: x['price'], reverse=True)[:5])
        top5_ask_qty = sum(level['quantity'] for level in sorted(ask_data, key=lambda x: x['price'])[:5])
        
        return {
            'timestamp': datetime.now(),
            'best_bid': best_bid['price'],
            'best_ask': best_ask['price'],
            'spread': spread,
            'spread_pct': spread_pct,
            'mid_price': mid_price,
            'total_bid_qty': total_bid_qty,
            'total_ask_qty': total_ask_qty,
            'total_bid_orders': total_bid_orders,
            'total_ask_orders': total_ask_orders,
            'qty_imbalance': qty_imbalance,
            'order_imbalance': order_imbalance,
            'top5_bid_qty': top5_bid_qty,
            'top5_ask_qty': top5_ask_qty,
        }
    
    def _detect_large_orders(self, security_id: str, bid_data: List[Dict], ask_data: List[Dict], timestamp: datetime) -> List[Dict]:
        """Detect unusually large orders"""
        alerts = []
        threshold = ALERT_CONFIG['large_order_threshold']
        
        # Check bids
        for level in bid_data:
            if level['quantity'] >= threshold:
                alerts.append({
                    'type': 'LARGE_BID_ORDER',
                    'security_id': security_id,
                    'timestamp': timestamp,
                    'price': level['price'],
                    'quantity': level['quantity'],
                    'orders': level['orders'],
                    'severity': 'HIGH' if level['quantity'] >= threshold * 2 else 'MEDIUM',
                    'message': f"Large bid order: {level['quantity']} @ {level['price']}"
                })
        
        # Check asks
        for level in ask_data:
            if level['quantity'] >= threshold:
                alerts.append({
                    'type': 'LARGE_ASK_ORDER',
                    'security_id': security_id,
                    'timestamp': timestamp,
                    'price': level['price'],
                    'quantity': level['quantity'],
                    'orders': level['orders'],
                    'severity': 'HIGH' if level['quantity'] >= threshold * 2 else 'MEDIUM',
                    'message': f"Large ask order: {level['quantity']} @ {level['price']}"
                })
        
        return alerts
    
    def _detect_order_imbalance(self, security_id: str, metrics: Dict, timestamp: datetime) -> List[Dict]:
        """Detect significant order flow imbalance"""
        alerts = []
        if not metrics:
            return alerts
            
        qty_imbalance = abs(metrics['qty_imbalance'])
        order_imbalance = abs(metrics['order_imbalance'])
        threshold = 1 / ALERT_CONFIG['imbalance_ratio_threshold']  # Convert ratio to imbalance threshold
        
        if qty_imbalance > threshold:
            direction = "BUY" if metrics['qty_imbalance'] > 0 else "SELL"
            alerts.append({
                'type': 'ORDER_IMBALANCE',
                'security_id': security_id,
                'timestamp': timestamp,
                'direction': direction,
                'imbalance_ratio': metrics['qty_imbalance'],
                'severity': 'HIGH' if qty_imbalance > threshold * 1.5 else 'MEDIUM',
                'message': f"Strong {direction} imbalance: {qty_imbalance:.2%} quantity imbalance"
            })
        
        return alerts
    
    def _detect_spread_compression(self, security_id: str, metrics: Dict, timestamp: datetime) -> List[Dict]:
        """Detect spread compression"""
        alerts = []
        if not metrics or security_id not in self.price_history:
            return alerts
            
        current_spread_pct = metrics['spread_pct']
        
        # Need more history for reliable signal
        recent_spreads = [m.get('spread_pct', 0) for m in list(self.price_history[security_id])[-15:] if m.get('spread_pct', 0) > 0]
        if len(recent_spreads) < 10:
            return alerts
            
        # Calculate trend in spread compression
        avg_spread = np.mean(recent_spreads[:-3])  # Exclude last 3 for comparison
        recent_avg = np.mean(recent_spreads[-3:])  # Last 3 periods
        
        if avg_spread == 0:
            return alerts
            
        compression_ratio = (avg_spread - recent_avg) / avg_spread
        
        if compression_ratio > ALERT_CONFIG['spread_compression_threshold']:
            alerts.append({
                'type': 'SPREAD_COMPRESSION',
                'security_id': security_id,
                'timestamp': timestamp,
                'current_spread': current_spread_pct,
                'avg_spread': avg_spread,
                'compression_ratio': compression_ratio,
                'severity': 'HIGH',
                'message': f"Spread compressing: {compression_ratio:.1%} tighter"
            })
        
        return alerts
    
    def _detect_iceberg_orders(self, security_id: str, bid_data: List[Dict], ask_data: List[Dict], timestamp: datetime) -> List[Dict]:
        """Detect potential iceberg orders"""
        alerts = []
        # Simplified iceberg detection - look for consistent replenishment at same price level
        # This is a basic implementation and would need more sophisticated logic for production
        return alerts
    
    def _detect_volume_spikes(self, security_id: str, metrics: Dict, timestamp: datetime) -> List[Dict]:
        """Detect volume spikes compared to recent average"""
        alerts = []
        if not metrics or security_id not in self.volume_history:
            return alerts
            
        current_total_qty = metrics['total_bid_qty'] + metrics['total_ask_qty']
        
        # Calculate recent average volume
        recent_volumes = [m.get('total_volume', 0) for m in list(self.volume_history[security_id])[-20:]]
        if len(recent_volumes) < 10:
            return alerts
            
        avg_volume = np.mean(recent_volumes)
        if avg_volume == 0:
            return alerts
            
        volume_ratio = current_total_qty / avg_volume
        
        if volume_ratio >= ALERT_CONFIG['volume_spike_threshold']:
            alerts.append({
                'type': 'VOLUME_SPIKE',
                'security_id': security_id,
                'timestamp': timestamp,
                'current_volume': current_total_qty,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'severity': 'HIGH' if volume_ratio >= ALERT_CONFIG['volume_spike_threshold'] * 1.5 else 'MEDIUM',
                'message': f"Volume spike: {volume_ratio:.1f}x average volume"
            })
        
        return alerts
    
    def _detect_level_breaks(self, security_id: str, bid_data: List[Dict], ask_data: List[Dict], timestamp: datetime) -> List[Dict]:
        """Detect when price breaks through multiple levels"""
        alerts = []
        if security_id not in self.last_depth:
            return alerts
            
        last_depth = self.last_depth[security_id]
        if not last_depth or 'bid' not in last_depth or 'ask' not in last_depth:
            return alerts
            
        # Get current and previous best prices
        current_best_bid = max(bid_data, key=lambda x: x['price'])['price'] if bid_data else 0
        current_best_ask = min(ask_data, key=lambda x: x['price'])['price'] if ask_data else float('inf')
        
        prev_best_bid = max(last_depth['bid'], key=lambda x: x['price'])['price'] if last_depth['bid'] else 0
        prev_best_ask = min(last_depth['ask'], key=lambda x: x['price'])['price'] if last_depth['ask'] else float('inf')
        
        # Count levels broken through
        if current_best_bid > prev_best_ask:  # Price moved up through ask levels
            levels_broken = len([level for level in last_depth['ask'] if level['price'] <= current_best_bid])
            if levels_broken >= ALERT_CONFIG['price_level_break_threshold']:
                alerts.append({
                    'type': 'UPSIDE_BREAKOUT',
                    'security_id': security_id,
                    'timestamp': timestamp,
                    'levels_broken': levels_broken,
                    'new_price': current_best_bid,
                    'severity': 'HIGH',
                    'message': f"Upside breakout: Broke through {levels_broken} ask levels"
                })
        
        elif current_best_ask < prev_best_bid:  # Price moved down through bid levels
            levels_broken = len([level for level in last_depth['bid'] if level['price'] >= current_best_ask])
            if levels_broken >= ALERT_CONFIG['price_level_break_threshold']:
                alerts.append({
                    'type': 'DOWNSIDE_BREAKOUT',
                    'security_id': security_id,
                    'timestamp': timestamp,
                    'levels_broken': levels_broken,
                    'new_price': current_best_ask,
                    'severity': 'HIGH',
                    'message': f"Downside breakout: Broke through {levels_broken} bid levels"
                })
        
        return alerts
    
    def _detect_order_flow_momentum(self, security_id: str, metrics: Dict, timestamp: datetime) -> List[Dict]:
        """Detect sustained order flow momentum"""
        alerts = []
        if not metrics or security_id not in self.order_flow_history:
            return alerts
            
        # Calculate momentum score based on recent imbalances
        recent_imbalances = [m.get('qty_imbalance', 0) for m in list(self.order_flow_history[security_id])[-10:]]
        if len(recent_imbalances) < 5:
            return alerts
            
        # Momentum = consistency of direction * magnitude
        avg_imbalance = np.mean(recent_imbalances)
        consistency = len([x for x in recent_imbalances if np.sign(x) == np.sign(avg_imbalance)]) / len(recent_imbalances)
        momentum_score = abs(avg_imbalance) * consistency
        
        if momentum_score >= ALERT_CONFIG['order_flow_momentum_threshold']:
            direction = "BULLISH" if avg_imbalance > 0 else "BEARISH"
            alerts.append({
                'type': 'ORDER_FLOW_MOMENTUM',
                'security_id': security_id,
                'timestamp': timestamp,
                'direction': direction,
                'momentum_score': momentum_score,
                'consistency': consistency,
                'avg_imbalance': avg_imbalance,
                'severity': 'HIGH',
                'message': f"{direction} momentum: {momentum_score:.2f} score, {consistency:.0%} consistency"
            })
        
        return alerts
    
    def _update_history(self, security_id: str, metrics: Dict):
        """Update historical data"""
        if not metrics:
            return
            
        # Add total volume for volume tracking
        metrics['total_volume'] = metrics['total_bid_qty'] + metrics['total_ask_qty']
        
        self.price_history[security_id].append(metrics)
        self.volume_history[security_id].append(metrics)
        self.order_flow_history[security_id].append(metrics)
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts sorted by timestamp"""
        return sorted(list(self.alerts)[-limit:], key=lambda x: x['timestamp'], reverse=True)
    
    def get_alerts_for_security(self, security_id: str, limit: int = 10) -> List[Dict]:
        """Get recent alerts for specific security"""
        security_alerts = [alert for alert in self.alerts if alert['security_id'] == security_id]
        return sorted(security_alerts[-limit:], key=lambda x: x['timestamp'], reverse=True)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary statistics of alerts"""
        if not self.alerts:
            return {
                'total_alerts': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'alert_types': {},
                'most_active_security': None
            }
        
        recent_alerts = list(self.alerts)[-50:]  # Last 50 alerts
        
        high_severity = len([a for a in recent_alerts if a.get('severity') == 'HIGH'])
        medium_severity = len([a for a in recent_alerts if a.get('severity') == 'MEDIUM'])
        
        # Count alert types
        alert_types = {}
        security_counts = {}
        
        for alert in recent_alerts:
            alert_type = alert['type']
            security_id = alert['security_id']
            
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            security_counts[security_id] = security_counts.get(security_id, 0) + 1
        
        most_active_security = max(security_counts.items(), key=lambda x: x[1])[0] if security_counts else None
        
        return {
            'total_alerts': len(recent_alerts),
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'alert_types': alert_types,
            'most_active_security': most_active_security,
            'last_alert_time': recent_alerts[-1]['timestamp'] if recent_alerts else None
        }

    def detect_institutional_patterns(self, security_id: str, bid_data: List[Dict], 
                                    ask_data: List[Dict], timestamp: datetime) -> List[Dict]:
        """Detect institutional trading patterns"""
        if not hasattr(self, 'institutional_detector'):
            self.institutional_detector = EnhancedInstitutionalDetection(self)
        
        return self.institutional_detector.detect_institutional_activity(
            security_id, bid_data, ask_data, timestamp
        )
    
class EnhancedInstitutionalDetection:
    """Enhanced detection for institutional vs retail trading patterns"""
    
    def __init__(self, alert_engine):
        self.alert_engine = alert_engine
        self.lot_sizes = {
            # Add more instruments and their lot sizes as needed
            'nifty': 75,
            'banknifty': 25,
            'default': 1
        }
    
    def detect_institutional_activity(self, security_id: str, bid_data: List[Dict], 
                                    ask_data: List[Dict], timestamp: datetime) -> List[Dict]:
        """Main institutional detection function"""
        alerts = []
        
        # Determine lot size for this security (you might want to maintain a mapping)
        lot_size = self._get_lot_size(security_id)
        
        # Analyze bid side
        bid_analysis = self._analyze_order_structure(bid_data, lot_size, "BID")
        alerts.extend(self._generate_institutional_alerts(security_id, bid_analysis, timestamp, "BID"))
        
        # Analyze ask side
        ask_analysis = self._analyze_order_structure(ask_data, lot_size, "ASK")
        alerts.extend(self._generate_institutional_alerts(security_id, ask_analysis, timestamp, "ASK"))
        
        # Cross-side analysis
        alerts.extend(self._analyze_cross_side_patterns(security_id, bid_analysis, ask_analysis, timestamp))
        
        return alerts
    
    def _get_lot_size(self, security_id: str) -> int:
        """Get lot size for security - you can enhance this with a proper mapping"""
        # Simple heuristic - you should maintain a proper instrument master
        if 'nifty' in security_id.lower() or security_id in ['26009', '26000']:
            return 75
        elif 'bank' in security_id.lower() or security_id in ['26009']:  # BankNifty token
            return 25
        return 1  # For equity or unknown instruments
    
    def _analyze_order_structure(self, levels: List[Dict], lot_size: int, side: str) -> Dict:
        """Analyze order structure to identify institutional patterns"""
        if not levels:
            return {}
        
        total_quantity = sum(level['quantity'] for level in levels)
        total_orders = sum(level['orders'] for level in levels)
        
        if total_orders == 0:
            return {}
        
        # Order size analysis
        order_sizes = []
        lot_aligned_orders = 0
        institutional_indicators = []
        
        for level in levels:
            if level['orders'] > 0:
                avg_order_size = level['quantity'] / level['orders']
                order_sizes.append(avg_order_size)
                
                # Check if orders are lot-aligned (typical retail behavior)
                if abs(avg_order_size % lot_size) < (lot_size * 0.1):  # Within 10% of lot size
                    lot_aligned_orders += level['orders']
                
                # Single large order indicators
                if level['orders'] == 1 and level['quantity'] >= INSTITUTIONAL_DETECTION_CONFIG['institutional_single_order_threshold']:
                    institutional_indicators.append({
                        'type': 'single_large_order',
                        'price': level['price'],
                        'quantity': level['quantity'],
                        'lot_multiple': level['quantity'] / lot_size
                    })
                
                # Block deal detection
                if level['quantity'] >= INSTITUTIONAL_DETECTION_CONFIG['block_deal_threshold']:
                    institutional_indicators.append({
                        'type': 'block_deal_size',
                        'price': level['price'],
                        'quantity': level['quantity'],
                        'orders': level['orders'],
                        'lot_multiple': level['quantity'] / lot_size
                    })
        
        # Calculate metrics
        avg_order_size = np.mean(order_sizes) if order_sizes else 0
        order_size_variance = np.var(order_sizes) if len(order_sizes) > 1 else 0
        retail_fragmentation_ratio = lot_aligned_orders / total_orders if total_orders > 0 else 0
        
        # Concentration analysis - check if few orders control large quantity
        levels_by_qty = sorted(levels, key=lambda x: x['quantity'], reverse=True)
        top_3_quantity = sum(level['quantity'] for level in levels_by_qty[:3])
        concentration_ratio = top_3_quantity / total_quantity if total_quantity > 0 else 0
        
        return {
            'side': side,
            'total_quantity': total_quantity,
            'total_orders': total_orders,
            'avg_order_size': avg_order_size,
            'order_size_variance': order_size_variance,
            'retail_fragmentation_ratio': retail_fragmentation_ratio,
            'concentration_ratio': concentration_ratio,
            'institutional_indicators': institutional_indicators,
            'lot_size': lot_size
        }
    
    def _generate_institutional_alerts(self, security_id: str, analysis: Dict, 
                                     timestamp: datetime, side: str) -> List[Dict]:
        """Generate alerts based on institutional pattern analysis"""
        alerts = []
        
        if not analysis:
            return alerts
        
        # Single large order detection
        for indicator in analysis.get('institutional_indicators', []):
            if indicator['type'] == 'single_large_order':
                alerts.append({
                    'type': 'INSTITUTIONAL_SINGLE_ORDER',
                    'security_id': security_id,
                    'timestamp': timestamp,
                    'side': side,
                    'price': indicator['price'],
                    'quantity': indicator['quantity'],
                    'lot_multiple': indicator['lot_multiple'],
                    'severity': 'HIGH',
                    'message': f"Institutional single order: {indicator['quantity']} ({indicator['lot_multiple']:.1f} lots) on {side} @ {indicator['price']}"
                })
            
            elif indicator['type'] == 'block_deal_size':
                alerts.append({
                    'type': 'BLOCK_DEAL_DETECTED',
                    'security_id': security_id,
                    'timestamp': timestamp,
                    'side': side,
                    'price': indicator['price'],
                    'quantity': indicator['quantity'],
                    'orders': indicator['orders'],
                    'lot_multiple': indicator['lot_multiple'],
                    'severity': 'HIGH',
                    'message': f"Block deal size: {indicator['quantity']} ({indicator['lot_multiple']:.0f} lots) in {indicator['orders']} orders on {side}"
                })
        
        # High concentration with low fragmentation (institutional characteristic)
        concentration = analysis.get('concentration_ratio', 0)
        fragmentation = analysis.get('retail_fragmentation_ratio', 1)
        
        if (concentration >= INSTITUTIONAL_DETECTION_CONFIG['institutional_order_concentration_threshold'] and 
            fragmentation <= (1 - INSTITUTIONAL_DETECTION_CONFIG['retail_fragmentation_threshold'])):
            
            alerts.append({
                'type': 'INSTITUTIONAL_CONCENTRATION',
                'security_id': security_id,
                'timestamp': timestamp,
                'side': side,
                'concentration_ratio': concentration,
                'fragmentation_ratio': fragmentation,
                'total_quantity': analysis['total_quantity'],
                'total_orders': analysis['total_orders'],
                'severity': 'MEDIUM',
                'message': f"Institutional pattern on {side}: {concentration:.1%} quantity concentration, low retail fragmentation"
            })
        
        # High order size variance (mixed institutional and retail)
        variance = analysis.get('order_size_variance', 0)
        avg_size = analysis.get('avg_order_size', 0)
        
        if avg_size > 0 and variance > (avg_size * INSTITUTIONAL_DETECTION_CONFIG['order_size_variance_threshold']):
            alerts.append({
                'type': 'MIXED_PLAYER_ACTIVITY',
                'security_id': security_id,
                'timestamp': timestamp,
                'side': side,
                'order_size_variance': variance,
                'avg_order_size': avg_size,
                'severity': 'MEDIUM',
                'message': f"Mixed player activity on {side}: High order size variance indicating both retail and institutional presence"
            })
        
        return alerts
    
    def _analyze_cross_side_patterns(self, security_id: str, bid_analysis: Dict, 
                                   ask_analysis: Dict, timestamp: datetime) -> List[Dict]:
        """Analyze patterns across bid and ask sides"""
        alerts = []
        
        if not bid_analysis or not ask_analysis:
            return alerts
        
        # Institutional vs retail dominance on different sides
        bid_institutional = (bid_analysis.get('concentration_ratio', 0) > 0.3 and 
                           bid_analysis.get('retail_fragmentation_ratio', 1) < 0.3)
        ask_institutional = (ask_analysis.get('concentration_ratio', 0) > 0.3 and 
                           ask_analysis.get('retail_fragmentation_ratio', 1) < 0.3)
        
        if bid_institutional and not ask_institutional:
            alerts.append({
                'type': 'INSTITUTIONAL_BID_DOMINANCE',
                'security_id': security_id,
                'timestamp': timestamp,
                'bid_concentration': bid_analysis.get('concentration_ratio', 0),
                'ask_concentration': ask_analysis.get('concentration_ratio', 0),
                'severity': 'HIGH',
                'message': "Institutional dominance on bid side vs retail on ask side - potential accumulation"
            })
        
        elif ask_institutional and not bid_institutional:
            alerts.append({
                'type': 'INSTITUTIONAL_ASK_DOMINANCE',
                'security_id': security_id,
                'timestamp': timestamp,
                'bid_concentration': bid_analysis.get('concentration_ratio', 0),
                'ask_concentration': ask_analysis.get('concentration_ratio', 0),
                'severity': 'HIGH',
                'message': "Institutional dominance on ask side vs retail on bid side - potential distribution"
            })
        
        # Simultaneous large orders on both sides (possible arbitrage or hedging)
        bid_large_orders = len([i for i in bid_analysis.get('institutional_indicators', []) 
                              if i['type'] in ['single_large_order', 'block_deal_size']])
        ask_large_orders = len([i for i in ask_analysis.get('institutional_indicators', []) 
                              if i['type'] in ['single_large_order', 'block_deal_size']])
        
        if bid_large_orders > 0 and ask_large_orders > 0:
            alerts.append({
                'type': 'BILATERAL_INSTITUTIONAL_ACTIVITY',
                'security_id': security_id,
                'timestamp': timestamp,
                'bid_large_orders': bid_large_orders,
                'ask_large_orders': ask_large_orders,
                'severity': 'HIGH',
                'message': f"Institutional activity on both sides - possible arbitrage/hedging ({bid_large_orders} bid, {ask_large_orders} ask large orders)"
            })
        
        return alerts

# ---------------------------
# Original classes with Alert Engine integration
# ---------------------------

def parse_header_slice(header_bytes: bytes):
    try:
        return struct.unpack('<hBBiI', header_bytes)
    except Exception:
        return None

def parse_depth_message(raw: bytes):
    if len(raw) < 12:
        return None

    header = parse_header_slice(raw[0:12])
    if not header:
        return None

    msg_length = header[0]
    msg_code = header[1]
    exchange_segment = header[2]
    security_id = header[3]

    body = raw[12:]
    packet_fmt = '<dII'
    packet_size = struct.calcsize(packet_fmt)
    depth = []
    
    for i in range(20):
        start = i * packet_size
        end = start + packet_size
        if end > len(body):
            break
        try:
            price, qty, orders = struct.unpack(packet_fmt, body[start:end])
        except struct.error:
            break
        depth.append({"price": float(price), "quantity": int(qty), "orders": int(orders)})
    
    if msg_code == 41:
        mtype = "Bid"
    elif msg_code == 51:
        mtype = "Ask"
    else:
        mtype = f"Other({msg_code})"

    return {
        "msg_length": msg_length,
        "msg_code": msg_code,
        "exchange_segment": exchange_segment,
        "security_id": security_id,
        "type": mtype,
        "depth": depth
    }

class DepthManager:
    def __init__(self, client_id: str, access_token: str, out_queue: queue.Queue, alert_engine: AlertEngine):
        self.client_id = client_id
        self.access_token = access_token
        self.out_queue = out_queue
        self.alert_engine = alert_engine

        self._instruments: List[Tuple[int, str]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.ws = None
        self.connected = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

    async def _shutdown(self):
        try:
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass
            await asyncio.sleep(0.1)
            self._loop.stop()
        except Exception:
            pass

    def _run_loop_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            tasks = asyncio.all_tasks(loop=self._loop)
            for t in tasks:
                t.cancel()
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()

    async def _main(self):
        while not self._stop_event.is_set():
            try:
                url = f"{DEPTH_WSS}?token={self.access_token}&clientId={self.client_id}&authType=2"
                async with websockets.connect(url, max_size=None) as ws:
                    self.ws = ws
                    self.connected = True
                    if self._instruments:
                        await self._send_subscribe(self._instruments)
                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=20)
                        except asyncio.TimeoutError:
                            try:
                                await ws.ping()
                            except Exception:
                                break
                            continue
                        if raw is None:
                            break
                        if isinstance(raw, str):
                            try:
                                j = json.loads(raw)
                                self.out_queue.put({"type": "text", "data": j})
                                continue
                            except Exception:
                                continue
                        if isinstance(raw, (bytes, bytearray)):
                            cursor = 0
                            while cursor < len(raw):
                                if cursor + 12 > len(raw):
                                    break
                                header_slice = raw[cursor:cursor+12]
                                parsed = parse_header_slice(header_slice)
                                if not parsed:
                                    break
                                msg_length = parsed[0]
                                if msg_length <= 0:
                                    break
                                end_idx = cursor + msg_length
                                if end_idx > len(raw):
                                    break
                                message_raw = raw[cursor:end_idx]
                                parsed_msg = parse_depth_message(message_raw)
                                if parsed_msg:
                                    self.out_queue.put(parsed_msg)
                                cursor = end_idx
                        else:
                            continue
            except Exception as e:
                self.out_queue.put({"type": "error", "data": str(e)})
                self.connected = False
                await asyncio.sleep(2)
        self.connected = False

    async def _send_subscribe(self, instruments: List[Tuple[int, str]]):
        if not self.ws:
            return
        exchange_map = {1: "NSE_EQ", 2: "NSE_FNO"}
        msg = {
            "RequestCode": DEPTH_20_REQ,
            "InstrumentCount": len(instruments),
            "InstrumentList": [
                {"ExchangeSegment": exchange_map.get(ex, str(ex)), "SecurityId": token}
                for ex, token in instruments
            ]
        }
        try:
            await self.ws.send(json.dumps(msg))
            self.out_queue.put({"type": "info", "data": f"Subscribed: {len(instruments)} instruments"})
        except Exception as e:
            self.out_queue.put({"type": "error", "data": f"Subscribe failed: {e}"})

    def subscribe(self, instruments: List[Tuple[int, str]]):
        existing = set(self._instruments)
        to_add = []
        for tup in instruments:
            if tup not in existing:
                existing.add(tup)
                to_add.append(tup)
        self._instruments = list(existing)
        if self._loop and self._loop.is_running() and to_add:
            asyncio.run_coroutine_threadsafe(self._send_subscribe(to_add), self._loop)

    def unsubscribe(self, instruments: List[Tuple[int, str]]):
        existing = set(self._instruments)
        for tup in instruments:
            existing.discard(tup)
        self._instruments = list(existing)
        self.out_queue.put({"type": "info", "data": f"Unsubscribed {len(instruments)} locally."})

# ---------------------------
# Enhanced Streamlit UI
# ---------------------------

def init_session():
    if "depth_manager" not in st.session_state:
        st.session_state["depth_queue"] = queue.Queue()
        st.session_state["alert_engine"] = AlertEngine()
        st.session_state["client_id"] = st.session_state.get("client_id", "1100244268")
        st.session_state["access_token"] = st.session_state.get("access_token", "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU5MDM3OTE2LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDI0NDI2OCJ9.NiJopQOOMl9I3sFI4HUp-d4a-9HKpEXo5EL6jtJheLsjubkzC1saXOx1mjIh-H8_IXCgIXqAjAYydE-sNBnKHg")
        st.session_state["depth_manager"] = DepthManager(
            client_id=st.session_state["client_id"],
            access_token=st.session_state["access_token"],
            out_queue=st.session_state["depth_queue"],
            alert_engine=st.session_state["alert_engine"]
        )
        st.session_state["subscribed"] = []
        st.session_state["latest"] = {}
        st.session_state["recent_alerts"] = []
        st.session_state["alert_sound"] = True

def add_subscription(exchange_int: int, security_id: str):
    tup = (int(exchange_int), str(security_id))
    st.session_state["subscribed"].append(tup)
    st.session_state["subscribed"] = list(dict.fromkeys(st.session_state["subscribed"]))
    st.session_state["depth_manager"].subscribe([tup])

def remove_subscription_tuple(tup):
    if tup in st.session_state["subscribed"]:
        st.session_state["subscribed"].remove(tup)
        st.session_state["depth_manager"].unsubscribe([tup])

def consume_queue_and_update_latest(max_messages=200):
    q: queue.Queue = st.session_state["depth_queue"]
    alert_engine: AlertEngine = st.session_state["alert_engine"]
    cnt = 0
    new_alerts = []
    
    while not q.empty() and cnt < max_messages:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        cnt += 1
        
        if msg.get("type") == "text":
            continue
        if msg.get("type") == "error":
            st.session_state.setdefault("_errors", []).append(msg["data"])
            continue
        if msg.get("type") in ("info",):
            st.session_state.setdefault("_infos", []).append(msg["data"])
            continue
            
        # Normal depth message - run alert analysis
        sec = str(msg["security_id"])
        meta = st.session_state["latest"].get(sec, {
            "bid": None, 
            "ask": None, 
            "exchange_segment": msg.get("exchange_segment")
        })
        
        if msg["type"] == "Bid":
            meta["bid"] = msg["depth"]
        elif msg["type"] == "Ask":
            meta["ask"] = msg["depth"]
        else:
            continue
            
        meta["last_ts"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["latest"][sec] = meta
        
        # Run alert analysis if we have both bid and ask data
        if meta.get("bid") and meta.get("ask"):
            alerts = alert_engine.analyze_depth_and_generate_alerts(sec, meta["bid"], meta["ask"])
            new_alerts.extend(alerts)
    
    # Update recent alerts
    if new_alerts:
        st.session_state["recent_alerts"] = alert_engine.get_recent_alerts(50)

def build_combined_df_for_security(sec_id: str):
    """
    Combine up to 20-level bid and ask into a DataFrame with columns:
    bid_price, bid_qty, bid_orders, ask_price, ask_qty, ask_orders
    """
    from itertools import zip_longest
    
    rec = st.session_state["latest"].get(str(sec_id))
    if not rec:
        return pd.DataFrame()
    bids = rec.get("bid") or []
    asks = rec.get("ask") or []
    
    # Sort bids desc, asks asc
    bids_sorted = sorted([b for b in bids if b["price"] > 0], key=lambda x: x["price"], reverse=True)
    asks_sorted = sorted([a for a in asks if a["price"] > 0], key=lambda x: x["price"])
    
    rows = []
    for b, a in zip_longest(bids_sorted, asks_sorted, fillvalue=None):
        row = {
            "bid_price": (b["price"] if b else None),
            "bid_qty": (b["quantity"] if b else None),
            "bid_orders": (b["orders"] if b else None),
            "ask_price": (a["price"] if a else None),
            "ask_qty": (a["quantity"] if a else None),
            "ask_orders": (a["orders"] if a else None),
        }
        rows.append(row)
    return pd.DataFrame(rows)

def create_depth_chart(sec_id: str):
    """Create interactive depth visualization"""
    rec = st.session_state["latest"].get(str(sec_id))
    if not rec or not rec.get("bid") or not rec.get("ask"):
        return None
        
    bids = sorted([b for b in rec["bid"] if b["price"] > 0], key=lambda x: x["price"], reverse=True)
    asks = sorted([a for a in rec["ask"] if a["price"] > 0], key=lambda x: x["price"])
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Bid side (green)
    bid_prices = [b["price"] for b in bids]
    bid_qtys = [b["quantity"] for b in bids]
    bid_cumulative = np.cumsum(bid_qtys)
    
    fig.add_trace(
        go.Scatter(
            x=bid_prices, 
            y=bid_cumulative,
            mode='lines+markers',
            name='Bid Depth',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        )
    )
    
    # Ask side (red)
    ask_prices = [a["price"] for a in asks]
    ask_qtys = [a["quantity"] for a in asks]
    ask_cumulative = np.cumsum(ask_qtys)
    
    fig.add_trace(
        go.Scatter(
            x=ask_prices, 
            y=ask_cumulative,
            mode='lines+markers',
            name='Ask Depth',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        )
    )
    
    fig.update_layout(
        title=f"Market Depth - {sec_id}",
        xaxis_title="Price",
        yaxis_title="Cumulative Quantity",
        height=400
    )
    
    return fig

def render_enhanced_depth_view(sec_id: str):
    """Render enhanced depth analysis and metrics"""
    rec = st.session_state["latest"].get(str(sec_id))
    if not rec or not rec.get("bid") or not rec.get("ask"):
        return None
        
    bids = rec.get("bid", [])
    asks = rec.get("ask", [])
    
    if not bids or not asks:
        return None
    
    # Calculate enhanced metrics
    best_bid = max(bids, key=lambda x: x['price'])
    best_ask = min(asks, key=lambda x: x['price'])
    
    # Volume metrics
    total_bid_qty = sum(level['quantity'] for level in bids)
    total_ask_qty = sum(level['quantity'] for level in asks)
    total_bid_orders = sum(level['orders'] for level in bids)
    total_ask_orders = sum(level['orders'] for level in asks)
    
    # Top 5 levels analysis
    top5_bids = sorted(bids, key=lambda x: x['price'], reverse=True)[:5]
    top5_asks = sorted(asks, key=lambda x: x['price'])[:5]
    
    top5_bid_qty = sum(level['quantity'] for level in top5_bids)
    top5_ask_qty = sum(level['quantity'] for level in top5_asks)
    
    # Imbalance calculations
    qty_imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) if (total_bid_qty + total_ask_qty) > 0 else 0
    order_imbalance = (total_bid_orders - total_ask_orders) / (total_bid_orders + total_ask_orders) if (total_bid_orders + total_ask_orders) > 0 else 0
    
    # Display metrics
    st.subheader("Enhanced Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Bid Quantity", f"{total_bid_qty:,}")
        st.metric("Total Bid Orders", f"{total_bid_orders:,}")
        st.metric("Top 5 Bid Qty", f"{top5_bid_qty:,}")
        st.metric("Avg Bid Size", f"{total_bid_qty/total_bid_orders:.1f}" if total_bid_orders > 0 else "0")
    
    with col2:
        st.metric("Total Ask Quantity", f"{total_ask_qty:,}")
        st.metric("Total Ask Orders", f"{total_ask_orders:,}")
        st.metric("Top 5 Ask Qty", f"{top5_ask_qty:,}")
        st.metric("Avg Ask Size", f"{total_ask_qty/total_ask_orders:.1f}" if total_ask_orders > 0 else "0")
    
    # Imbalance analysis
    st.subheader("Order Flow Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        imbalance_color = "normal"
        if abs(qty_imbalance) > 0.3:
            imbalance_color = "inverse" if qty_imbalance < 0 else "normal"
            
        st.metric(
            "Quantity Imbalance", 
            f"{qty_imbalance:.2%}",
            help="Positive = More bids, Negative = More asks"
        )
    
    with col2:
        st.metric(
            "Order Count Imbalance", 
            f"{order_imbalance:.2%}",
            help="Positive = More bid orders, Negative = More ask orders"
        )
    
    # Market depth table
    st.subheader("Market Depth Table")
    df = build_combined_df_for_security(sec_id)
    if not df.empty:
        st.dataframe(
            df.head(20),  # Show top 10 levels
            use_container_width=True,
            hide_index=True
        )
    
    return {
        'total_bid_qty': total_bid_qty,
        'total_ask_qty': total_ask_qty,
        'qty_imbalance': qty_imbalance,
        'order_imbalance': order_imbalance
    }

def render_alert_panel():
    """Render the alert panel with recent alerts"""
    st.header("Smart Alerts")
    
    # Alert configuration
    with st.expander("Alert Settings"):
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Large Order Threshold", 
                          value=ALERT_CONFIG['large_order_threshold'], 
                          key="large_order_threshold")
            st.number_input("Imbalance Ratio Threshold", 
                          value=ALERT_CONFIG['imbalance_ratio_threshold'], 
                          key="imbalance_ratio_threshold")
            st.number_input("Volume Spike Threshold", 
                          value=ALERT_CONFIG['volume_spike_threshold'], 
                          key="volume_spike_threshold")
        with col2:
            st.number_input("Spread Compression %", 
                          value=ALERT_CONFIG['spread_compression_threshold'], 
                          key="spread_compression_threshold")
            st.number_input("Iceberg Detection Count", 
                          value=ALERT_CONFIG['iceberg_detection_threshold'], 
                          key="iceberg_detection_threshold")
            st.checkbox("Enable Alert Sound", value=st.session_state.get("alert_sound", True), key="alert_sound")
    
    # Recent alerts
    recent_alerts = st.session_state.get("recent_alerts", [])
    
    if recent_alerts:
        st.subheader("Recent Alerts")
        
        # Alert summary stats
        high_severity = len([a for a in recent_alerts if a.get('severity') == 'HIGH'])
        medium_severity = len([a for a in recent_alerts if a.get('severity') == 'MEDIUM'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts", len(recent_alerts))
        col2.metric("High Severity", high_severity)
        col3.metric("Medium Severity", medium_severity)
        
        # Alert list
        for i, alert in enumerate(recent_alerts[:10]):  # Show last 10 alerts
            severity_color = "🔴" if alert.get('severity') == 'HIGH' else "🟡"
            timestamp_str = alert['timestamp'].strftime("%H:%M:%S")
            
            alert_type = alert['type'].replace('_', ' ').title()
            
            with st.container():
                if alert.get('severity') == 'HIGH':
                    st.error(f"{severity_color} **{alert_type}** - {alert['security_id']} at {timestamp_str}")
                else:
                    st.warning(f"{severity_color} **{alert_type}** - {alert['security_id']} at {timestamp_str}")
                
                st.caption(alert['message'])
                
                # Additional details based on alert type
                if alert['type'] in ['LARGE_BID_ORDER', 'LARGE_ASK_ORDER']:
                    st.caption(f"Price: {alert['price']}, Quantity: {alert['quantity']}, Orders: {alert['orders']}")
                elif alert['type'] == 'ORDER_IMBALANCE':
                    st.caption(f"Direction: {alert['direction']}, Imbalance: {alert['imbalance_ratio']:.2%}")
                elif alert['type'] == 'VOLUME_SPIKE':
                    st.caption(f"Volume Ratio: {alert['volume_ratio']:.1f}x average")
                # ADD THESE NEW LINES HERE:
                elif alert['type'] == 'INSTITUTIONAL_SINGLE_ORDER':
                    st.caption(f"Side: {alert['side']}, Price: {alert['price']}, Quantity: {alert['quantity']}, Lots: {alert['lot_multiple']:.1f}")
                elif alert['type'] == 'BLOCK_DEAL_DETECTED':
                    st.caption(f"Side: {alert['side']}, Price: {alert['price']}, Quantity: {alert['quantity']}, Orders: {alert['orders']}")
                elif alert['type'] == 'INSTITUTIONAL_CONCENTRATION':
                    st.caption(f"Side: {alert['side']}, Concentration: {alert['concentration_ratio']:.1%}")
                elif alert['type'] in ['INSTITUTIONAL_BID_DOMINANCE', 'INSTITUTIONAL_ASK_DOMINANCE']:
                    st.caption(f"Bid Conc: {alert['bid_concentration']:.1%}, Ask Conc: {alert['ask_concentration']:.1%}")
                elif alert['type'] == 'BILATERAL_INSTITUTIONAL_ACTIVITY':
                    st.caption(f"Bid Orders: {alert['bid_large_orders']}, Ask Orders: {alert['ask_large_orders']}")
                elif alert['type'] == 'MIXED_PLAYER_ACTIVITY':
                    st.caption(f"Side: {alert['side']}, Avg Order Size: {alert['avg_order_size']:.0f}")
                
                st.divider()
    else:
        st.info("No alerts yet. Start the feed and subscribe to instruments to receive alerts.")

def main():
    st.set_page_config(
        page_title="Smart Market Depth Dashboard", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    init_session()

    st.title("Smart Market Depth Dashboard with AI Alerts")
    
    # Top navigation
    tab1, tab2, tab3 = st.tabs(["Market Depth", "Alerts", "Controls"])
    
    with tab3:  # Controls tab
        st.header("Connection & Subscription Controls")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Connection controls
            st.subheader("Connection")
            start_btn = st.button("Start Feed", type="primary")
            stop_btn = st.button("Stop Feed")
            
            connected = st.session_state["depth_manager"].connected
            status_color = "🟢" if connected else "🔴"
            st.write(f"Status: {status_color} {'Connected' if connected else 'Disconnected'}")
            
            if start_btn:
                st.session_state["depth_manager"].start()
                st.success("Depth manager started.")
                time.sleep(1)
                st.rerun()
                
            if stop_btn:
                st.session_state["depth_manager"].stop()
                st.success("Depth manager stopped.")
                time.sleep(1)
                st.rerun()
        
        with col2:
            # Subscription controls
            st.subheader("Add Subscription")
            st.caption("Exchange codes: 1 = NSE_EQ, 2 = NSE_FNO")
            
            col2a, col2b, col2c = st.columns([1, 2, 1])
            with col2a:
                ex = st.number_input("Exchange", min_value=1, max_value=99, step=1, value=2)
            with col2b:
                sec_id = st.text_input("Security ID", value="", placeholder="e.g., 26009 for Nifty")
            with col2c:
                add_btn = st.button("Add")
            
            if add_btn:
                if sec_id.strip() == "":
                    st.warning("Enter a Security ID before subscribing.")
                else:
                    add_subscription(ex, sec_id.strip())
                    st.success(f"Subscribed to ({ex}, {sec_id.strip()})")
                    time.sleep(1)
                    st.rerun()
        
        # Current subscriptions
        st.subheader("Current Subscriptions")
        if st.session_state["subscribed"]:
            for i, tup in enumerate(st.session_state["subscribed"]):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"Exchange: {tup[0]}")
                with col2:
                    st.write(f"Security ID: {tup[1]}")
                with col3:
                    if st.button("🗑️", key=f"unsub_{i}", help="Unsubscribe"):
                        remove_subscription_tuple(tup)
                        st.rerun()
        else:
            st.info("No active subscriptions")
        
        # Logs
        if st.session_state.get("_errors"):
            st.subheader("Errors")
            for error in st.session_state["_errors"][-3:]:
                st.error(error)
                
        if st.session_state.get("_infos"):
            st.subheader("Info")
            for info in st.session_state["_infos"][-3:]:
                st.info(info)

    with tab2:  # Alerts tab
        render_alert_panel()

    with tab1:  # Market Depth tab
        st.header("Live Market Depth")
        
        # Process incoming messages and generate alerts
        consume_queue_and_update_latest()
        
        # Display depth for each subscribed security
        if not st.session_state["subscribed"]:
            st.info("Subscribe to at least one instrument in the Controls tab to view market depth")
        else:
            # Create tabs for each security
            security_tabs = st.tabs([f"{tup[1]}" for tup in st.session_state["subscribed"]])
            
            for idx, tup in enumerate(st.session_state["subscribed"]):
                sec = str(tup[1])
                with security_tabs[idx]:
                    meta = st.session_state["latest"].get(sec)
                    
                    if not meta:
                        st.warning("Waiting for data...")
                        continue
                        
                    if not (meta.get("bid") and meta.get("ask")):
                        st.warning("Waiting for complete bid/ask data...")
                        continue
                    
                    # Security header with key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    best_bid = max(meta["bid"], key=lambda x: x["price"]) if meta["bid"] else None
                    best_ask = min(meta["ask"], key=lambda x: x["price"]) if meta["ask"] else None
                    
                    if best_bid and best_ask:
                        spread = best_ask["price"] - best_bid["price"]
                        mid_price = (best_bid["price"] + best_ask["price"]) / 2
                        spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
                        
                        col1.metric("Best Bid", f"₹{best_bid['price']:.2f}", f"{best_bid['quantity']} qty")
                        col2.metric("Best Ask", f"₹{best_ask['price']:.2f}", f"{best_ask['quantity']} qty")
                        col3.metric("Spread", f"₹{spread:.2f}", f"{spread_pct:.3f}%")
                        col4.metric("Mid Price", f"₹{mid_price:.2f}")
                    
                    # Recent alerts for this security
                    security_alerts = st.session_state["alert_engine"].get_alerts_for_security(sec, 3)
                    if security_alerts:
                        st.subheader("Recent Alerts")
                        for alert in security_alerts:
                            alert_color = "error" if alert.get('severity') == 'HIGH' else "warning"
                            getattr(st, alert_color)(f"{alert['type'].replace('_', ' ').title()}: {alert['message']}")
                    
                    # Depth visualization and data
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Depth Chart")
                        chart = create_depth_chart(sec)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.info("Chart will appear once data is available")
                    
                    with col2:
                        enhanced_metrics = render_enhanced_depth_view(sec)
                    
                    st.caption(f"Last updated: {meta.get('last_ts', 'Never')}")

    # Auto refresh
    time.sleep(0.5)
    st.rerun()

if __name__ == "__main__":
    main()
