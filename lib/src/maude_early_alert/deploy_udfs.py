"""
Snowflake UDF 배포 스크립트
- udfs/ 폴더의 SQL 파일들을 Snowflake에 등록
- AWS Secrets Manager로 인증
"""

import sys
from pathlib import Path
from typing import Dict

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException

from maude_early_alert.utils.secrets import get_secret


class UDFDeployer:
    """UDF SQL 파일들을 Snowflake에 배포"""

    def __init__(
        self,
        udfs_dir: str = "udfs",
        secret_name: str = 'snowflake/bronze/credentials',
        region_name: str = 'ap-northeast-2'
    ):
        self.udfs_dir = Path(udfs_dir)
        self.secret_name = secret_name
        self.region_name = region_name
        self.session = self._get_session()
        self.results: Dict[str, Dict] = {}
    
    def _get_session(self) -> Session:
        """AWS Secrets Manager에서 Snowpark 세션 생성"""
        secret = get_secret(self.secret_name, self.region_name)
        return Session.builder.configs(secret).create()
    
    def _validate_udfs_dir(self) -> None:
        """UDF 디렉토리 존재 확인"""
        if not self.udfs_dir.exists():
            raise FileNotFoundError(
                f"❌ UDF 디렉토리를 찾을 수 없어요: {self.udfs_dir}\n"
                f"   udfs/ 폴더를 먼저 만들어주세요!"
            )
        
        sql_files = list(self.udfs_dir.glob("*.sql"))
        if not sql_files:
            raise FileNotFoundError(
                f"⚠️  {self.udfs_dir}에 SQL 파일이 없어요!\n"
                f"   UDF 정의 SQL 파일을 추가해주세요."
            )
    
    def _read_sql_file(self, filepath: Path) -> str:
        """SQL 파일 읽기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # UTF-8 실패 시 다른 인코딩 시도
            with open(filepath, 'r', encoding='cp949') as f:
                return f.read().strip()
    
    def _deploy_single_udf(self, sql_file: Path) -> Dict:
        """단일 UDF SQL 파일 실행"""
        filename = sql_file.name

        try:
            print(f"📝 {filename} 실행 중...")

            sql = self._read_sql_file(sql_file)

            if not sql:
                return {
                    'status': 'skipped',
                    'message': 'SQL 파일이 비어있어요'
                }

            self.session.sql(sql).collect()
            
            print(f"   ✅ {filename} 배포 완료!")
            return {
                'status': 'success',
                'message': '배포 성공'
            }
            
        except SnowparkSQLException as e:
            error_msg = str(e)
            print(f"   ❌ {filename} 실패: {error_msg}")
            return {
                'status': 'failed',
                'message': error_msg,
                'error_code': e.error_code if hasattr(e, 'error_code') else None
            }
        
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ {filename} 실패: {error_msg}")
            return {
                'status': 'failed',
                'message': error_msg
            }
    
    def deploy_all(self) -> Dict[str, Dict]:
        """모든 UDF 배포"""
        print("=" * 60)
        print("🚀 Snowflake UDF 배포 시작")
        print("=" * 60)
        
        # UDF 디렉토리 검증
        self._validate_udfs_dir()
        
        # SQL 파일 목록
        sql_files = sorted(self.udfs_dir.glob("*.sql"))
        print(f"\n📋 발견된 SQL 파일: {len(sql_files)}개\n")
        
        # 각 파일 실행
        for sql_file in sql_files:
            result = self._deploy_single_udf(sql_file)
            self.results[sql_file.name] = result
        
        # 결과 요약
        self._print_summary()
        
        return self.results
    
    def _print_summary(self) -> None:
        """배포 결과 요약 출력"""
        success_count = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed_count = sum(1 for r in self.results.values() if r['status'] == 'failed')
        skipped_count = sum(1 for r in self.results.values() if r['status'] == 'skipped')
        
        print("\n" + "=" * 60)
        print("📊 배포 결과 요약")
        print("=" * 60)
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {failed_count}개")
        print(f"⏭️  스킵: {skipped_count}개")
        print(f"📝 총 파일: {len(self.results)}개")
        
        # 실패한 파일 상세 출력
        if failed_count > 0:
            print("\n⚠️  실패한 파일 상세:")
            for filename, result in self.results.items():
                if result['status'] == 'failed':
                    print(f"   - {filename}")
                    print(f"     → {result['message'][:100]}...")
        
        print("=" * 60)
        
        # 실패가 있으면 exit code 1
        if failed_count > 0:
            print("\n⚠️  일부 UDF 배포에 실패했어요!")
            sys.exit(1)
        else:
            print("\n🎉 모든 UDF 배포 완료!")
    
    def close(self) -> None:
        """세션 종료"""
        self.session.close()


def main():
    udfs_dir = Path(__file__).absolute().parent / 'udfs'
    """메인 실행 함수"""
    deployer = UDFDeployer(
        udfs_dir=str(udfs_dir),
        secret_name='snowflake/udf/credentials'
    )
    
    try:
        deployer.deploy_all()
    finally:
        deployer.close()


if __name__ == "__main__":
    main()