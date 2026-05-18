import PageHeader   from '../components/PageHeader'
import BulletinCard from '../components/BulletinCard'

export default function BulletinPage({ bulletin, latest }) {
  return (
    <div className="fade-up">
      <PageHeader title="Intelligence Bulletin" subtitle="AI-generated market stress memo · 5 requests per hour" page="bulletin" regime={latest?.data?.regime} regimeLabel={latest?.data?.regime_label} />
      <div style={{ padding: '16px' }}>
        <BulletinCard data={bulletin.data} loading={bulletin.loading} error={bulletin.error} onRetry={bulletin.reload} />
      </div>
    </div>
  )
}
