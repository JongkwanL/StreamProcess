module github.com/streamprocess/streamprocess

go 1.21

require (
	github.com/go-redis/redis/v8 v8.11.5
	github.com/golang/protobuf v1.5.3
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.18.1
	github.com/prometheus/client_golang v1.17.0
	github.com/sirupsen/logrus v1.9.3
	github.com/spf13/cobra v1.8.0
	github.com/spf13/viper v1.18.2
	google.golang.org/grpc v1.59.0
	google.golang.org/protobuf v1.31.0
	k8s.io/api v0.28.4
	k8s.io/apimachinery v0.28.4
	k8s.io/client-go v0.28.4
	k8s.io/metrics v0.28.4
)

require (
	github.com/aws/aws-sdk-go v1.50.0
	github.com/otiai10/gosseract/v2 v2.4.1
	github.com/stretchr/testify v1.8.4
	gopkg.in/yaml.v3 v3.0.1
)